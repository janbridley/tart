//! Command-line interface: tart reads agent definitions from a TOML file,
//! named by `--agents` or the default under `~/.config/tart`.

use std::ffi::OsString;
use std::path::PathBuf;

use anyhow::bail;
use clap::{Arg, ArgMatches, Command};

/// The command line, parsed once: the agents file to load, and whether this session is chat.
pub(crate) struct Cli {
    /// The TOML file describing the available agents.
    pub(crate) agents: PathBuf,
    /// Whether `--chat` was passed: web tools only, no shell or filesystem.
    pub(crate) chat: bool,
    /// Whether `--gpu` was passed: Metal compute inside the sandbox.
    pub(crate) gpu: bool,
}

impl Cli {
    /// Parse the process's argv.
    pub(crate) fn parse() -> anyhow::Result<Self> {
        Self::from(&command().get_matches(), std::env::var_os("HOME"))
    }

    /// The command line as `matches` name it, with the agents file resolved
    /// against `home` or a nice error message.
    fn from(matches: &ArgMatches, home: Option<OsString>) -> anyhow::Result<Self> {
        Ok(Self {
            agents: resolve(matches, home)?,
            chat: matches.get_flag("chat"),
            gpu: matches.get_flag("gpu"),
        })
    }
}

/// The agents file `matches` selects: `--agents FILE` when given, else
/// `~/.config/tart/providers.toml`.
fn resolve(matches: &ArgMatches, home: Option<OsString>) -> anyhow::Result<PathBuf> {
    if let Some(file) = matches.get_one::<String>("agents") {
        return Ok(file.clone().into());
    }
    // The same `$HOME`-rooted `.config/tart` the session store uses.
    let home = home
        .map(PathBuf::from)
        .expect("$HOME is not set; nowhere to read the default providers.toml");
    let path = home.join(".config/tart/providers.toml");
    if !path.is_file() {
        bail!(
            "No agents file at {}\n  Set that configuration, or pass --agents FILE\n",
            path.display()
        );
    }
    Ok(path)
}

fn command() -> Command {
    Command::new("tart")
        .about("A terminal chat front end for the tart agent harness.")
        .arg(Arg::new("agents").long("agents").value_name("FILE").help(
            "TOML file describing the available agents [default: ~/.config/tart/providers.toml]",
        ))
        .arg(
            Arg::new("chat")
                .long("chat")
                .action(clap::ArgAction::SetTrue)
                .help("Chat mode: web tools only, no shell or filesystem access"),
        )
        .arg(Arg::new("gpu").long("gpu").action(clap::ArgAction::SetTrue).help(
            "Grant the coding sandbox GPU access for Metal compute (toggle at runtime with /gpu)",
        ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `--agents` names the file to read, without checking it exists:
    /// `Config::load` reports a missing named file with the path as typed.
    #[test]
    fn the_flag_names_the_agents_file() {
        let matches = command()
            .try_get_matches_from(["tart", "--agents", "no-such-file.toml"])
            .expect("the flag parses");

        let path = resolve(&matches, None).expect("the named file is used as-is");
        assert_eq!(path, PathBuf::from("no-such-file.toml"));
    }

    /// Without the flag, the file under `$HOME` is read when it exists.
    #[test]
    fn without_the_flag_the_home_default_is_read() {
        let matches = command()
            .try_get_matches_from(["tart"])
            .expect("no flag is required");

        // A scratch `$HOME` holding the default file.
        let home = tempfile::tempdir().expect("scratch $HOME");
        let config = home.path().join(".config/tart");
        std::fs::create_dir_all(&config).expect("scratch .config/tart");
        std::fs::write(config.join("providers.toml"), "").expect("scratch providers.toml");

        let path = resolve(&matches, Some(home.as_ref().as_os_str().to_owned()))
            .expect("the default file exists");
        assert_eq!(path, config.join("providers.toml"));
    }

    /// A missing default names the fix, unlike a missing named file.
    #[test]
    fn a_missing_default_names_the_fix() {
        let matches = command()
            .try_get_matches_from(["tart"])
            .expect("no flag is required");

        let home = tempfile::tempdir().expect("scratch $HOME");
        let error = resolve(&matches, Some(home.as_ref().as_os_str().to_owned()))
            .expect_err("the default file does not exist")
            .to_string();

        assert!(error.starts_with("No agents file at "), "{error}");
        assert!(
            error.contains(
                home.path()
                    .join(".config/tart/providers.toml")
                    .display()
                    .to_string()
                    .as_str()
            ),
            "{error}"
        );
        assert!(error.contains("--agents"), "{error}");
    }

    /// `--gpu` parses true, and its absence false, alongside the agents file.
    #[test]
    fn the_gpu_flag_parses() {
        let with = Cli::from(
            &command()
                .try_get_matches_from(["tart", "--agents", "f.toml", "--gpu"])
                .expect("the flags parse"),
            None,
        )
        .expect("the named file is used as-is");
        assert!(with.gpu);
        assert_eq!(with.agents, PathBuf::from("f.toml"));

        let without = Cli::from(
            &command()
                .try_get_matches_from(["tart", "--agents", "f.toml"])
                .expect("the flags parse"),
            None,
        )
        .expect("the named file is used as-is");
        assert!(!without.gpu);
    }

    /// `--chat` parses true, and its absence false, alongside the agents file.
    #[test]
    fn the_chat_flag_parses() {
        let with = Cli::from(
            &command()
                .try_get_matches_from(["tart", "--agents", "f.toml", "--chat"])
                .expect("the flags parse"),
            None,
        )
        .expect("the named file is used as-is");
        assert!(with.chat);
        assert_eq!(with.agents, PathBuf::from("f.toml"));

        let without = Cli::from(
            &command()
                .try_get_matches_from(["tart", "--agents", "f.toml"])
                .expect("the flags parse"),
            None,
        )
        .expect("the named file is used as-is");
        assert!(!without.chat);
    }
}
