//! The `--agents` TOML file: providers, their credentials, and their agents.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, bail};
use itertools::Itertools;
use serde::Deserialize;
use tart_agents::ReasoningEffort;

/// A parsed agents file. `default_agent` picks the agent the TUI runs.
#[derive(Debug)]
pub(crate) struct Config {
    default_agent: DefaultAgent,
    /// Provider tables by their TOML key, sorted for deterministic errors.
    providers: BTreeMap<String, Provider>,
}

/// The agent the TUI wires into its loop, with its API key resolved.
#[derive(Debug)]
pub(crate) struct ResolvedAgent {
    /// The provider's display name; the table key when no label is set.
    pub(crate) provider: String,
    pub(crate) name: String,
    pub(crate) base_url: String,
    pub(crate) api_key: String,
    pub(crate) model: String,
    pub(crate) effort: Option<ReasoningEffort>,
    /// Extra system prompt, appended after the built-in one.
    pub(crate) instructions: Option<String>,
    /// The model's context window, for the status line's token gauge.
    pub(crate) context_tokens: Option<u64>,
}

/// The status-line label, e.g. "z.ai · tart".
impl fmt::Display for ResolvedAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} · {}", self.provider, self.name)
    }
}

/// One provider table: a base URL, a credential, and the agents behind it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Provider {
    /// Display label; defaults to the table key.
    label: Option<String>,
    base_url: String,
    api_key: Auth,
    #[serde(default)]
    agents: Vec<AgentSpec>,
}

/// How a provider's API key is obtained.
///
/// The type in the TOML file lets us resolve both api keys and shell strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Auth {
    /// An environment variable name, written as a bare string.
    Env(String),
    /// A command that writes the key to stdout, written as an array.
    Command(Vec<String>),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DefaultAgent {
    provider: String,
    name: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentSpec {
    name: String,
    model: String,
    reasoning_effort: Option<ReasoningEffort>,
    instructions: Option<String>,
    context_tokens: Option<u64>,
}

impl Config {
    /// Read and parse the agents file, validating every provider in it.
    pub(crate) fn load(path: &Path) -> anyhow::Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("failed to parse {}", path.display()))
    }

    /// Parse and validate agent definitions; separate from `load` so tests
    /// need no files.
    pub(crate) fn parse(text: &str) -> anyhow::Result<Self> {
        // Two-phase: whole file to a Table first, so each provider's
        // deserialization error can name the table it came from.
        let mut table = toml::from_str::<toml::Table>(text)?;
        let default_agent = match table.remove("default_agent") {
            Some(value) => {
                DefaultAgent::deserialize(value).context("invalid [default_agent] section")?
            }
            None => bail!("missing required [default_agent] section"),
        };
        let mut providers = BTreeMap::new();
        for (key, value) in table {
            let provider = Provider::deserialize(value)
                .with_context(|| format!("invalid provider [{key}]"))?;
            if let Some(dup) = provider.agents.iter().map(|a| &a.name).duplicates().next() {
                bail!("provider [{key}] defines two agents named '{dup}'");
            }

            providers.insert(key, provider);
        }
        Ok(Self { default_agent, providers })
    }

    /// Resolve the default agent: look it up, obtain its API key.
    pub(crate) fn default_agent(&self) -> anyhow::Result<ResolvedAgent> {
        let key = &self.default_agent.provider;
        let provider = self.providers.get(key).ok_or_else(|| {
            let defined = self.providers.keys().map(String::as_str).join(", ");
            anyhow::anyhow!(
                "default_agent names provider '{key}', which is not defined (defined: {defined})"
            )
        })?;
        let target = &self.default_agent.name;
        let agent = provider
            .agents
            .iter()
            .find(|a| &a.name == target)
            .ok_or_else(|| {
                let has = provider.agents.iter().map(|a| a.name.as_str()).join(", ");
                anyhow::anyhow!("provider [{key}] has no agent named '{target}' (has: {has})")
            })?;

        Ok(ResolvedAgent {
            provider: provider.label.as_deref().unwrap_or(key).to_owned(),
            name: agent.name.clone(),
            base_url: provider.base_url.clone(),
            api_key: resolve_api_key(key, &provider.api_key)?,
            model: agent.model.clone(),
            effort: agent.reasoning_effort.clone(),
            instructions: agent.instructions.clone(),
            context_tokens: agent.context_tokens,
        })
    }
}

/// Obtain the provider's API key: the env variable or command in `api_key`.
fn resolve_api_key(key: &str, auth: &Auth) -> anyhow::Result<String> {
    match auth {
        Auth::Env(var) => std::env::var(var)
            .with_context(|| format!("provider [{key}]: environment variable {var} is not set")),
        Auth::Command(argv) => {
            let [program, extra_args @ ..] = argv.as_slice() else {
                bail!("provider [{key}]: api_key command is empty");
            };
            let command = argv.join(" ");
            let output = Command::new(program)
                .args(extra_args)
                .output()
                .with_context(|| format!("provider [{key}]: failed to run `{command}`"))?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let detail = match stderr.trim() {
                    "" => String::new(),
                    err => format!(": {err}"),
                };
                bail!(
                    "provider [{key}]: `{command}` failed with {}{detail}",
                    output.status
                );
            }
            let secret = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if secret.is_empty() {
                bail!("provider [{key}]: `{command}` printed nothing");
            }
            Ok(secret)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;

    /// The smallest file that resolves and includes every optional field.
    const MINIMAL: &str = r#"
[default_agent]
provider = "zai"
name = "tart"

[zai]
label = "z.ai"
base_url = "https://api.z.ai/api/v1"
api_key = ["echo", "secret-key"]

[[zai.agents]]
name = "tart"
model = "glm-5.3"
reasoning_effort = "high"
instructions = "Write performant, safe code."
"#;

    #[test]
    fn minimal_file_resolves_the_default_agent() {
        let agent = Config::parse(MINIMAL).unwrap().default_agent().unwrap();

        assert_eq!(agent.to_string(), "z.ai · tart");
        assert_eq!(agent.base_url, "https://api.z.ai/api/v1");
        assert_eq!(agent.api_key, "secret-key");
        assert_eq!(agent.model, "glm-5.3");
        assert_eq!(agent.effort, Some(ReasoningEffort::High));
        assert_eq!(
            agent.instructions.as_deref(),
            Some("Write performant, safe code.")
        );
    }

    #[test]
    fn label_falls_back_to_the_table_key() {
        let text = MINIMAL.replacen("\nlabel = \"z.ai\"\nbase_url", "\nbase_url", 1);
        let agent = Config::parse(&text).unwrap().default_agent().unwrap();

        assert_eq!(agent.to_string(), "zai · tart");
    }

    #[test]
    fn missing_default_agent_section_is_an_error() {
        let text = "[zai]\nbase_url = \"https://api.z.ai/api/v1\"\napi_key = \"PATH\"\n";
        let error = Config::parse(text).unwrap_err().to_string();

        assert!(error.contains("default_agent"), "{error}");
    }

    #[test]
    fn unknown_provider_in_default_agent_is_an_error() {
        let text = MINIMAL.replacen("provider = \"zai\"\nname", "provider = \"nope\"\nname", 1);
        let error = Config::parse(&text)
            .unwrap()
            .default_agent()
            .unwrap_err()
            .to_string();

        assert!(error.contains("nope"), "{error}");
        assert!(error.contains("zai"), "{error}");
    }

    #[test]
    fn unknown_agent_name_is_an_error() {
        let text = MINIMAL.replacen("name = \"tart\"\n\n[zai]", "name = \"Typo\"\n\n[zai]", 1);
        let error = Config::parse(&text)
            .unwrap()
            .default_agent()
            .unwrap_err()
            .to_string();

        assert!(error.contains("Typo"), "{error}");
        assert!(error.contains("tart"), "{error}");
    }

    #[test]
    fn missing_api_key_is_an_error() {
        let text = MINIMAL.replacen("\napi_key = [\"echo\", \"secret-key\"]\n", "\n", 1);
        // The missing field surfaces as a serde cause; walk the chain.
        let error = format!("{:#}", Config::parse(&text).unwrap_err());

        assert!(error.contains("api_key"), "{error}");
    }

    #[test]
    fn empty_command_is_an_error() {
        let text = MINIMAL.replacen("api_key = [\"echo\", \"secret-key\"]", "api_key = []", 1);
        let error = Config::parse(&text)
            .unwrap()
            .default_agent()
            .unwrap_err()
            .to_string();

        assert!(error.contains("command is empty"), "{error}");
    }

    #[test]
    fn failing_command_reports_its_exit() {
        let text =
            MINIMAL.replacen("api_key = [\"echo\", \"secret-key\"]", "api_key = [\"false\"]", 1);
        let error = Config::parse(&text)
            .unwrap()
            .default_agent()
            .unwrap_err()
            .to_string();

        assert!(error.contains("failed with"), "{error}");
    }

    #[test]
    fn silent_command_is_an_error() {
        let text =
            MINIMAL.replacen("api_key = [\"echo\", \"secret-key\"]", "api_key = [\"true\"]", 1);
        let error = Config::parse(&text)
            .unwrap()
            .default_agent()
            .unwrap_err()
            .to_string();

        assert!(error.contains("printed nothing"), "{error}");
    }

    #[test]
    fn bad_reasoning_effort_is_an_error() {
        let text = MINIMAL.replacen("reasoning_effort = \"high\"", "reasoning_effort = \"max\"", 1);
        // `{:#}` walks anyhow's cause chain; `to_string` shows only the context.
        let error = format!("{:#}", Config::parse(&text).unwrap_err());

        assert!(error.contains("max"), "{error}");
    }

    #[test]
    fn unknown_agent_key_is_an_error() {
        let text = MINIMAL.replacen(
            "instructions = \"Write performant, safe code.\"",
            "instruction = \"Write performant, safe code.\"",
            1,
        );
        let error = format!("{:#}", Config::parse(&text).unwrap_err());

        assert!(error.contains("instruction"), "{error}");
    }

    #[test]
    fn context_tokens_round_trips() {
        let text = MINIMAL.replacen(
            "reasoning_effort = \"high\"",
            "reasoning_effort = \"high\"\ncontext_tokens = 200000",
            1,
        );
        let agent = Config::parse(&text).unwrap().default_agent().unwrap();

        assert_eq!(agent.context_tokens, Some(200_000));
    }

    #[test]
    fn duplicate_agent_names_are_an_error() {
        let text = format!("{MINIMAL}\n[[zai.agents]]\nname = \"tart\"\nmodel = \"glm-4-flash\"\n");
        let error = Config::parse(&text).unwrap_err().to_string();

        assert!(error.contains("two agents"), "{error}");
    }

    #[test]
    fn agents_list_is_optional() {
        let text = format!(
            "{MINIMAL}\n[deepseek]\nbase_url = \"https://api.deepseek.com\"\napi_key = \"DEEPSEEK_API_KEY\"\n"
        );

        // The unwraps are the assertion: `[deepseek]` parses without agents.
        Config::parse(&text).unwrap().default_agent().unwrap();
    }

    #[test]
    fn typoed_provider_table_is_named_in_the_error() {
        // `[deeseek...]` reads as a second, broken provider table.
        let text =
            format!("{MINIMAL}\n[deeseek.authentication]\nenv_variable = \"DEEPSEEK_API_KEY\"\n");
        let error = Config::parse(&text).unwrap_err().to_string();

        assert!(error.contains("deeseek"), "{error}");
    }

    #[test]
    fn env_variable_auth_reads_the_environment() {
        let text =
            MINIMAL.replacen("api_key = [\"echo\", \"secret-key\"]", "api_key = \"PATH\"", 1);
        let agent = Config::parse(&text).unwrap().default_agent().unwrap();

        assert_eq!(agent.api_key, std::env::var("PATH").unwrap());
    }

    #[test]
    fn missing_env_variable_is_an_error() {
        let text = MINIMAL.replacen(
            "api_key = [\"echo\", \"secret-key\"]",
            "api_key = \"TART_NO_SUCH_ENVIRONMENT_VARIABLE\"",
            1,
        );
        let error = Config::parse(&text)
            .unwrap()
            .default_agent()
            .unwrap_err()
            .to_string();

        assert!(error.contains("TART_NO_SUCH_ENVIRONMENT_VARIABLE"), "{error}");
    }
}
