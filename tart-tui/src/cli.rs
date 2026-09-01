//! Command-line interface: tart reads agent definitions from a TOML file,
//! named by `--agents` or the default under `~/.config/tart`.

use std::ffi::OsString;
use std::path::PathBuf;

use anyhow::bail;
use clap::{Arg, ArgMatches, Command};

/// The `--agents` command line, with a resolved path or nice error message..
pub(crate) fn agents_path() -> anyhow::Result<PathBuf> {
    resolve(&command().get_matches(), std::env::var_os("HOME"))
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
    Command::new("tart-tui")
        .about("A terminal chat front end for the tart agent harness.")
        .arg(Arg::new("agents").long("agents").value_name("FILE").help(
            "TOML file describing the available agents [default: ~/.config/tart/providers.toml]",
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
}
