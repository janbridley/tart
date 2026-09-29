//! Startup: the session the command line describes, and the pane over it.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use tart_agents::{
    Agent, CHAT_PROJECT, ChatMode, SESSIONS_ROOT, Session, Transcript, prompts, sandbox::Policy,
};

use crate::cli;
use crate::config;
use crate::pane::Pane;

const REFUSAL: &str = "cd into a project directory, or pass --chat for a session \
                   with no filesystem access";

/// Refuse the coding session when its writable root would cover the user's home dir.
fn refuse_bare_root(dir: &Path) -> anyhow::Result<()> {
    let canonical = std::fs::canonicalize(dir)
        .with_context(|| format!("failed to resolve the working directory: {}", dir.display()))?;
    if canonical == Path::new("/") {
        anyhow::bail!("refusing to run at the filesystem root: {REFUSAL}");
    }
    // `home.starts_with(&canonical)` is containment: the working directory is
    // home, or an ancestor of it.
    if let Some(home) = std::env::home_dir()
        && let Ok(home) = std::fs::canonicalize(&home)
        && home.starts_with(&canonical)
    {
        anyhow::bail!("tart should not be run in or above $HOME: {REFUSAL}");
    }
    Ok(())
}

/// The session kind the command line selects.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// The agentic default: the cwd's project, tools, and prompt.
    Coding,
    /// `--chat`: web tools only, the minimal prompt, the CHAT directory.
    Chat,
}

impl Kind {
    /// The kind `chat` selects.
    fn of(chat: bool) -> Self {
        if chat { Self::Chat } else { Self::Coding }
    }

    /// The agent this kind runs: the `[chat_agent]` pick for chat, the
    /// default agent for coding.
    fn agent(self, config: &config::Config) -> anyhow::Result<config::ResolvedAgent> {
        match self {
            Self::Coding => config.default_agent(),
            Self::Chat => config.chat_agent(),
        }
    }

    /// The mode the agent runs in.
    fn mode(self) -> ChatMode {
        match self {
            Self::Coding => ChatMode::Default,
            Self::Chat => ChatMode::Chat,
        }
    }

    /// Where this kind's sessions record and resume from.
    fn project(self, cwd: &Path) -> PathBuf {
        match self {
            // Chat sessions live under their own CHAT directory, not the cwd's.
            Self::Chat => PathBuf::from(CHAT_PROJECT),
            Self::Coding => cwd.to_path_buf(),
        }
    }

    /// The policy this kind of agent is built with. The coding root has already
    /// been screened by `refuse_bare_root` by the time this runs; `gpu` opts
    /// the coding sandbox into Metal compute.
    fn policy(self, cwd: &Path, gpu: bool) -> anyhow::Result<Policy> {
        match self {
            Self::Coding => {
                let policy = Policy::new(cwd)?.exclude_git();
                Ok(if gpu { policy.allow_gpu() } else { policy })
            }
            Self::Chat => Ok(Policy::no_access()),
        }
    }

    /// The transcript this kind opens: the agentic prompt, or chat's minimal one.
    fn transcript(self) -> anyhow::Result<Transcript> {
        match self {
            Self::Coding => Transcript::new(),
            Self::Chat => Transcript::new_with(prompts::CHAT),
        }
    }
}

/// What `run` needs, assembled once at startup.
pub(crate) struct Tui {
    /// The providers file, for `/model` picks.
    pub(crate) config: config::Config,
    /// The agent the session talks to.
    pub(crate) agent: Agent,
    /// Where the session records.
    pub(crate) session: Session,
    /// The conversation the pane renders.
    pub(crate) transcript: Transcript,
    /// The project directory this session records under.
    pub(crate) project: PathBuf,
    /// The status line's `provider · agent` label.
    pub(crate) label: String,
    /// The model's context window, for the status line's token gauge.
    pub(crate) context_tokens: Option<u64>,
    /// The session kind: the banner, and the pane's chat flag.
    kind: Kind,
}

/// The command line becomes the session it describes.
impl TryFrom<&cli::Cli> for Tui {
    type Error = anyhow::Error;

    fn try_from(cli: &cli::Cli) -> Result<Self, Self::Error> {
        let kind = Kind::of(cli.chat);
        // The coding grant root, fetched once and screened before anything else
        // loads, so a bare-cwd invocation reports its own error, not a config one.
        let cwd = std::env::current_dir()?;
        if kind == Kind::Coding {
            refuse_bare_root(&cwd)?;
        }
        let config = config::Config::load(&cli.agents)?;
        let agent_config = kind.agent(&config)?;
        let label = agent_config.to_string();
        let context_tokens = agent_config.context_tokens;
        let project = kind.project(&cwd);
        let mut agent = agent_config.into_agent(kind.policy(&cwd, cli.gpu)?);
        agent.set_mode(kind.mode());
        Ok(Self {
            session: Session::start(&SESSIONS_ROOT, &project),
            transcript: kind.transcript()?,
            config,
            agent,
            project,
            label,
            context_tokens,
            kind,
        })
    }
}

impl Tui {
    /// The pane over the setup: wired to its agent and conversation, its
    /// session directory for `/resume`, and a banner naming what it runs.
    pub(crate) fn open_pane(&self) -> Pane {
        let mut pane = Pane::default();
        pane.set_session_dir(SESSIONS_ROOT.clone(), self.project.clone());
        pane.set_chat(self.kind == Kind::Chat);
        pane.set_control(self.agent.handle());
        pane.set_conversation(&self.transcript);
        pane.note(match self.kind {
            Kind::Coding => format!("tart · {}", self.label),
            Kind::Chat => format!("tart · {} · chat", self.label),
        });
        pane.set_context_tokens(self.context_tokens);
        pane.set_models(self.config.agents());
        pane
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;

    /// A providers file whose key resolves without the environment.
    fn providers(dir: &std::path::Path) -> std::path::PathBuf {
        let file = dir.join("providers.toml");
        std::fs::write(
            &file,
            "[default_agent]\nprovider = \"zai\"\nname = \"a\"\n\
             \n[zai]\nbase_url = \"http://localhost:9\"\napi_key = [\"echo\", \"k\"]\n\
             \n[[zai.agents]]\nname = \"a\"\nmodel = \"m\"\n",
        )
        .unwrap();
        file
    }

    /// The chat command line builds a chat `Tui`: the agent in Chat mode, the
    /// CHAT project, and the minimal prompt opening the record.
    #[test]
    fn a_chat_cli_builds_a_chat_tui() {
        let dir = tempfile::tempdir().unwrap();
        let cli = cli::Cli {
            agents: providers(dir.path()),
            chat: true,
            gpu: false,
        };

        let tui = Tui::try_from(&cli).unwrap();

        assert_eq!(tui.agent.mode(), ChatMode::Chat);
        assert_eq!(tui.project, PathBuf::from(CHAT_PROJECT));
        let items = serde_json::to_value(tui.transcript.request_items()).unwrap();
        assert_eq!(items[0]["role"], "system");
        assert_eq!(items[0]["content"], prompts::CHAT);
    }

    /// The gpu flag reaches the coding policy, and only the coding policy:
    /// chat is no-access whatever the flag says.
    #[test]
    fn the_gpu_flag_wires_into_the_coding_policy() {
        let dir = tempfile::tempdir().unwrap();
        let gpu = Kind::Coding.policy(dir.path(), true).unwrap().render();
        assert!(gpu.contains("AGXDeviceUserClient"), "{gpu}");
        let plain = Kind::Coding.policy(dir.path(), false).unwrap().render();
        assert!(!plain.contains("AGXDeviceUserClient"), "{plain}");
        let chat = Kind::Chat.policy(dir.path(), true).unwrap().render();
        assert!(!chat.contains("AGXDeviceUserClient"), "{chat}");
    }

    #[test]
    fn coding_refuses_roots_covering_home() {
        let err = refuse_bare_root(Path::new("/")).unwrap_err().to_string();
        assert!(err.contains("filesystem root"), "{err}");

        let Some(home) = std::env::home_dir() else {
            return;
        };
        let home = std::fs::canonicalize(&home).unwrap();
        for root in [home.as_path(), home.parent().expect("home has a parent")] {
            let err = refuse_bare_root(root).unwrap_err().to_string();
            assert!(err.contains("in or above $HOME"), "{root:?}: {err}");
        }

        // A symlink to home resolves to home and is refused identically;
        // sibling branches of home and scratch directories are not its ancestors.
        let dir = tempfile::tempdir().unwrap();
        let link = dir.path().join("home-link");
        std::os::unix::fs::symlink(&home, &link).unwrap();
        assert!(refuse_bare_root(&link).is_err());
        if Path::new("/Volumes").is_dir() {
            refuse_bare_root(Path::new("/Volumes")).unwrap();
        }
        refuse_bare_root(dir.path()).unwrap();
    }

    /// The bare command line stays the coding kind.
    #[test]
    fn a_bare_cli_builds_a_coding_tui() {
        let dir = tempfile::tempdir().unwrap();
        let cli = cli::Cli {
            agents: providers(dir.path()),
            chat: false,
            gpu: false,
        };

        let tui = Tui::try_from(&cli).unwrap();

        assert_eq!(tui.agent.mode(), ChatMode::Default);
        assert_eq!(tui.project, std::env::current_dir().unwrap());
        let items = serde_json::to_value(tui.transcript.request_items()).unwrap();
        assert_ne!(items[0]["content"], prompts::CHAT);
    }
}
