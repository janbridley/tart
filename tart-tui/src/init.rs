//! Startup: the session the command line describes, and the pane over it.

use std::path::PathBuf;

use tart_agents::{
    Agent, CHAT_PROJECT, ChatMode, SESSIONS_ROOT, Session, Transcript, prompts, sandbox::Policy,
};

use crate::cli;
use crate::config;
use crate::pane::Pane;

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

    /// The mode the agent runs in.
    fn mode(self) -> ChatMode {
        match self {
            Self::Coding => ChatMode::Default,
            Self::Chat => ChatMode::Chat,
        }
    }

    /// Where this kind's sessions record and resume from.
    fn project(self) -> anyhow::Result<PathBuf> {
        match self {
            Self::Coding => Ok(std::env::current_dir()?),
            // Chat sessions live under their own CHAT directory, not the cwd's.
            Self::Chat => Ok(PathBuf::from(CHAT_PROJECT)),
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
        let config = config::Config::load(&cli.agents)?;
        let agent_config = config.default_agent()?;
        let label = agent_config.to_string();
        let context_tokens = agent_config.context_tokens;
        let kind = Kind::of(cli.chat);
        let project = kind.project()?;
        // The cwd-rooted policy serves the coding kinds; chat's tool calls run
        // under `Policy::none` instead (see `Agent::policy`), so the grant is
        // inert there.
        let policy = Policy::new(std::env::current_dir()?)?.exclude_git();
        let mut agent = agent_config.into_agent(policy);
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
