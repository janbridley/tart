//! `~/.config/tart/settings.toml`, for options that do not need a CLI flag.
//!
//! ```toml
//! ["agents.md"]
//! owners = ["janbridley"]
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

/// The parsed settings file, defaulting when missing or invalid.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Settings {
    /// The `["agents.md"]` table.
    #[serde(default, rename = "agents.md")]
    agents_md: AgentsMd,
}

/// The `["agents.md"]` table.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentsMd {
    /// Remote owners whose repositories' AGENTS.md loads.
    #[serde(default)]
    owners: Vec<String>,
}

impl Settings {
    /// Read `~/.config/tart/settings.toml`, or default when it is absent.
    pub(crate) fn load() -> Self {
        match std::env::var_os("HOME") {
            Some(home) => Self::at(&PathBuf::from(home).join(".config/tart/settings.toml")),
            None => Self::default(),
        }
    }

    /// Read `path`, defaulting on any error as settings are optional.
    #[allow(clippy::print_stderr, reason = "a bad settings file should warn")]
    fn at(path: &Path) -> Self {
        match fs::read_to_string(path).map(|text| toml::from_str::<Self>(&text)) {
            Ok(Ok(settings)) => settings,
            Err(_) => Self::default(),
            Ok(Err(error)) => {
                eprintln!("tart: ignoring invalid {}: {error}", path.display());
                Self::default()
            }
        }
    }

    /// The `AGENTS.md` in `cwd`, when this repository is allowed to load one.
    pub(crate) fn agents_md(&self, cwd: &Path) -> Option<String> {
        self.for_remote(origin_url(cwd).as_deref(), cwd)
    }

    /// [`Settings::agents_md`] with the remote supplied, so tests need no git.
    ///
    /// A repository without an `origin` remote is local work and always loads;
    /// with one, the remote's owner must be listed, and a URL too exotic to
    /// parse counts as unlisted. Symlinks are skipped: this is a harness-side
    /// read outside the sandbox, and an AGENTS.md should not point at files the
    /// model could not read itself.
    fn for_remote(&self, remote: Option<&str>, cwd: &Path) -> Option<String> {
        let listed = remote.map(owner_of).is_none_or(|owner| {
            owner.is_some_and(|owner| self.agents_md.owners.iter().any(|o| o == owner))
        });
        let path = cwd.join("AGENTS.md");
        listed
            .then(|| fs::symlink_metadata(&path).ok())
            .flatten()
            .filter(std::fs::Metadata::is_file)
            .and_then(|_| fs::read_to_string(&path).ok())
            .filter(|text| !text.trim().is_empty())
    }
}

/// The `origin` remote's URL, or `None` when there is no repository or remote.
fn origin_url(cwd: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["remote", "get-url", "origin"])
        .current_dir(cwd)
        .output()
        .ok()?;
    let url = String::from_utf8(output.stdout).ok()?;
    let url = url.trim();
    (output.status.success() && !url.is_empty()).then(|| url.to_string())
}

/// The first path segment of a remote URL past its host: `jenna` in both
/// `git@github.com:janbridley/tart.git` and `https://github.com/janbridley/tart`.
fn owner_of(url: &str) -> Option<&str> {
    let tail = url.split_once("://").map_or(url, |(_, tail)| tail);
    let after_host = match tail.split_once(':') {
        // scp-style `git@host:owner/repo`
        Some((host, rest)) if !host.contains('/') => rest,
        // URL-style `host/owner/repo`
        _ => tail.split_once('/')?.1,
    };
    after_host.split('/').next().filter(|owner| !owner.is_empty())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use super::*;

    /// Settings allowing one owner, as `["agents.md"]` writes it.
    fn allowing_jenna() -> Settings {
        toml::from_str("[\"agents.md\"]\nowners = [\"jenna\"]").unwrap()
    }

    /// A directory whose AGENTS.md holds `text`.
    fn dir_with(text: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), text).unwrap();
        dir
    }

    #[test]
    fn owner_of_reads_both_remote_forms() {
        assert_eq!(owner_of("git@github.com:jenna/tart.git"), Some("jenna"));
        assert_eq!(owner_of("https://github.com/jenna/tart"), Some("jenna"));
        assert_eq!(owner_of("ssh://git@github.com/jenna/tart.git"), Some("jenna"));
        assert_eq!(owner_of("https://github.com"), None);
        assert_eq!(owner_of("not a url"), None);
    }

    #[test]
    fn the_table_parses_and_rejects_unknown_fields() {
        assert_eq!(
            toml::from_str::<Settings>("[\"agents.md\"]")
                .unwrap()
                .agents_md
                .owners,
            [] as [String; 0]
        );
        assert!(toml::from_str::<Settings>("[agents_md]").is_err());
        assert!(toml::from_str::<Settings>("[\"agents.md\"]\nunknown = 1").is_err());
    }

    #[test]
    fn a_missing_or_invalid_file_is_the_default() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            Settings::at(&dir.path().join("settings.toml")).agents_md.owners,
            [] as [String; 0]
        );
        let path = dir.path().join("settings.toml");
        std::fs::write(&path, "not toml [").unwrap();
        assert_eq!(Settings::at(&path).agents_md.owners, [] as [String; 0]);
    }

    #[test]
    fn a_remote_free_repository_loads_unconditionally() {
        assert_eq!(
            allowing_jenna().for_remote(None, dir_with("hi").path()),
            Some("hi".to_string())
        );
    }

    #[test]
    fn a_remote_repository_needs_its_owner_listed() {
        let stranger = "https://github.com/stranger/x";
        let jenna = "git@github.com:jenna/tart.git";
        let unparseable = "git@github.com";
        assert_eq!(
            allowing_jenna().for_remote(Some(stranger), dir_with("hi").path()),
            None
        );
        assert_eq!(
            allowing_jenna().for_remote(Some(unparseable), dir_with("hi").path()),
            None
        );
        assert_eq!(
            allowing_jenna().for_remote(Some(jenna), dir_with("hi").path()),
            Some("hi".to_string())
        );
    }

    #[test]
    fn only_regular_files_load() {
        // A symlinked AGENTS.md is skipped, as is a directory of that name.
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("AGENTS.md"), "hi").unwrap();
        std::os::unix::fs::symlink(outside.path().join("AGENTS.md"), dir.path().join("AGENTS.md"))
            .unwrap();
        assert_eq!(allowing_jenna().for_remote(None, dir.path()), None);

        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("AGENTS.md")).unwrap();
        assert_eq!(allowing_jenna().for_remote(None, dir.path()), None);
    }

    #[test]
    fn empty_files_load_as_none() {
        assert_eq!(allowing_jenna().for_remote(None, dir_with("  \n").path()), None);
    }

    #[test]
    fn a_real_repository_decides_by_its_remote() {
        let dir = tempfile::tempdir().unwrap();
        let git = |args: &[&str]| {
            let ok = std::process::Command::new("git")
                .args(args)
                .current_dir(dir.path())
                .status()
                .unwrap()
                .success();
            assert!(ok, "git {args:?} failed");
        };
        git(&["init"]);
        std::fs::write(dir.path().join("AGENTS.md"), "hi").unwrap();
        // No origin yet: local work, so the file loads.
        assert_eq!(allowing_jenna().agents_md(dir.path()), Some("hi".to_string()));
        // Cloned from a stranger: skipped.
        git(&["remote", "add", "origin", "https://github.com/stranger/x"]);
        assert_eq!(allowing_jenna().agents_md(dir.path()), None);
    }
}
