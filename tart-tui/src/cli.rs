//! Command-line interface: tart requires a TOML file of agent definitions.

use std::path::PathBuf;

use clap::{Arg, Command};

/// The `--agents` command line, or clap's error output and a nonzero exit.
pub(crate) fn agents_path() -> PathBuf {
    command()
        .get_matches()
        .get_one::<String>("agents")
        .expect("`agents` is required")
        .clone()
        .into()
}

fn command() -> Command {
    Command::new("tart-tui")
        .about("A terminal chat front end for the tart agent harness.")
        .arg(
            Arg::new("agents")
                .long("agents")
                .value_name("FILE")
                .required(true)
                .help("TOML file describing the available agents"),
        )
}
