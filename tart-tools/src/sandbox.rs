//! Run commands inside the macOS Seatbelt sandbox.
//!
//! A [`Policy`] allows for rust-native configuration of `sandbox-exec` policies for
//! efficient sandboxing. [`Policy::command`] returns a [`std::process::Command`] that
//! runs the program under that policy:
//!
//! ```
//! use tart_tools::sandbox::Policy;
//! # fn main() -> anyhow::Result<()> {
//! let policy = Policy::new(std::env::current_dir()?)?.exclude_git();
//! let out = policy.command("echo").arg("hello!").output()?;
//! assert!(out.status.success());
//! assert_eq!(String::from_utf8_lossy(&out.stdout), "hello!\n");
//!
//! // The cleared environment can be extended as usual:
//! let var_value = "env_hello!";
//! let out = policy
//!     .command("printenv")
//!     .arg("GREETING")
//!     .env("GREETING", &var_value)
//!     .output()?;
//! assert!(out.status.success());
//! assert_eq!(String::from_utf8_lossy(&out.stdout).trim_end(), var_value);
//! # Ok(())
//! # }
//! ```
//!
//! The module is built for running commands requested by a model, so it is
//! deny-by-default beyond the filesystem grants:
//!
//! - Network access is denied (the base profile is `(deny default)`).
//! - The child environment is cleared except for a minimal `PATH` and the granted temp
//!   directory, so secrets held by the caller cannot be printed into captured output.
//!   Re-add variables with the usual `std` methods.
//! - Standard input is [`Stdio::null`](std::process::Stdio::null) by default, and the
//!   policy denies reads and writes on `/dev/tty` and the `/dev/ttys*` devices.
//! - Binaries outside the platform read baseline (for example `/opt/homebrew/bin`)
//!   cannot be executed unless their directory is granted with
//!   [`Policy::add_read_only_root`].
//!
//! Known limits of the vendored platform defaults, which this module cannot
//! remove: writes are allowed under the system temp trees (`/tmp`,
//! `/private/tmp`, `/var/tmp`) regardless of the granted roots — so a
//! workspace inside a temp tree is not write-restricted. Exclusions are still
//! honored there: each is emitted as an explicit `(deny file-write* …)`, which
//! outranks the platform-default allows. And because Seatbelt filters match
//! paths, not inodes, a hard link created inside a writable root can route
//! writes into an excluded file; treat exclusions as tamper-evident, not
//! tamper-proof.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result};

/// Hardcoded path to the sandbox executable
const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

/// The only variables the child environment is seeded with after clearing.
const SANDBOXED_PATH: &str = "/usr/bin:/bin:/usr/sbin:/sbin";

// Vendored from openai/codex `codex-rs/sandboxing/src` at commit
// be6e8eac029b183056b7e4402879f15d2c85f61b (v0.147.0), Apache-2.0
const BASE_POLICY: &str = include_str!("sbpl/seatbelt_base_policy.sbpl");
const PLATFORM_DEFAULTS: &str = include_str!("sbpl/restricted_read_only_platform_defaults.sbpl");

/// A macOS Seatbelt sandbox profile under which commands can be run.
///
/// The profile grants write access to a set of canonicalized roots (the
/// environment temp directory is included when resolvable), read access to
/// everything the vendored platform defaults cover, and nothing else the
/// module docs do not already disclaim. Build one with [`Policy::new`], then
/// run commands via [`Policy::command`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Policy {
    /// Paths that are *writable* in the sandboxed environment.
    writable: Vec<PathBuf>,
    read_only: Vec<PathBuf>,
    /// Subpaths (e.g. `.git`) that are kept read-only inside every writable root.
    excluded: Vec<OsString>,
    /// The canonicalized temp directory granted as a writable root, if available.
    ///
    /// This path becomes `TMPDIR` in the sandboxed environment.
    temp: Option<PathBuf>,
}

/// The rendered profile plus its `-D` parameter bindings.
#[derive(Debug)]
struct CompiledPolicy {
    text: String,
    params: Vec<(String, OsString)>,
}

impl Policy {
    /// Creates a policy granting write access to `writable_root` and `TMPDIR`.
    ///
    /// # Errors
    ///
    /// Returns an error when `writable_root` does not exist or cannot be
    /// canonicalized. An unresolvable temp directory is skipped rather than
    /// failing the policy.
    #[inline]
    pub fn new<P: AsRef<Path>>(writable_root: P) -> Result<Self> {
        let mut policy = Self {
            writable: vec![canonicalize_root(writable_root.as_ref())?],
            read_only: Vec::new(),
            excluded: Vec::new(),
            temp: None,
        };
        if let Some(temp) = std::env::var_os("TMPDIR").filter(|temp| !temp.is_empty())
            && let Ok(canonical) = canonicalize_root(Path::new(&temp))
        {
            if !policy.writable.contains(&canonical) {
                policy.writable.push(canonical.clone());
            }
            policy.temp = Some(canonical);
        }
        Ok(policy)
    }

    /// Grant write access to another existing directory.
    ///
    /// Duplicate paths are ignored.
    ///
    /// # Errors
    ///
    /// Returns an error when `root` does not exist, cannot be canonicalized, or
    /// overlaps a read-only root.
    #[inline]
    pub fn add_writable_root<P: AsRef<Path>>(mut self, root: P) -> Result<Self> {
        let canonical = canonicalize_root(root.as_ref())?;
        if let Some(read_only) = self
            .read_only
            .iter()
            .find(|read_only| read_only.starts_with(&canonical) || canonical.starts_with(read_only))
        {
            anyhow::bail!(
                "writable root {} overlaps read-only root {}",
                canonical.display(),
                read_only.display()
            );
        }
        if !self.writable.contains(&canonical) {
            self.writable.push(canonical);
        }
        Ok(self)
    }

    /// Grants read-only access to another existing directory
    ///
    /// # Errors
    ///
    /// Returns an error when `root` does not exist, cannot be canonicalized, or
    /// overlaps a writable root.
    #[inline]
    pub fn add_read_only_root<P: AsRef<Path>>(mut self, root: P) -> Result<Self> {
        let canonical = canonicalize_root(root.as_ref())?;
        if let Some(writable) = self
            .writable
            .iter()
            .find(|writable| writable.starts_with(&canonical) || canonical.starts_with(writable))
        {
            anyhow::bail!(
                "read-only root {} overlaps writable root {}",
                canonical.display(),
                writable.display()
            );
        }
        if !self.read_only.contains(&canonical) {
            self.read_only.push(canonical);
        }
        Ok(self)
    }

    /// Keep a relative subpath read-only inside every writable root.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty, absolute, or root-escaping path; to
    /// exclude one specific location, join it onto the root before calling.
    #[inline]
    pub fn exclude<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
        let relative = path.as_ref();
        let raw = relative.as_os_str();
        anyhow::ensure!(!raw.is_empty(), "exclusion path is empty");
        anyhow::ensure!(
            relative.is_relative(),
            "exclusion {} is absolute",
            relative.display()
        );
        // Count depth so `..` may only cancel components it stays under.
        let mut depth = 0usize;
        for component in relative.components() {
            match component {
                std::path::Component::CurDir => {}
                std::path::Component::ParentDir => {
                    anyhow::ensure!(
                        depth > 0,
                        "exclusion {} escapes the writable root",
                        relative.display()
                    );
                    depth -= 1;
                }
                _ => depth += 1,
            }
        }
        anyhow::ensure!(
            depth > 0,
            "exclusion {} resolves to a writable root itself",
            relative.display()
        );
        let relative = raw.to_owned();
        if !self.excluded.contains(&relative) {
            self.excluded.push(relative);
        }
        Ok(self)
    }

    /// Convenience for [`Policy::exclude`] with `.git`.
    #[must_use]
    #[inline]
    pub fn exclude_git(mut self) -> Self {
        let git = OsString::from(".git");
        if !self.excluded.contains(&git) {
            self.excluded.push(git);
        }
        self
    }

    /// The complete SBPL profile text, exactly as passed to `sandbox-exec -p`.
    ///
    /// Granted paths appear only as `(param "NAME")` references; their values
    /// travel separately as `-DNAME=value` argv elements (see
    /// [`Policy::command`]).
    #[must_use]
    #[inline]
    pub fn render(&self) -> String {
        self.compile().text
    }

    /// A [`Command`] for `program` that runs under this policy:
    /// `sandbox-exec -p <render()> -DNAME=value ... -- <program>`.
    ///
    /// The command starts from a cleared environment (a minimal `PATH`, plus `TMPDIR`)
    /// for the granted temp directory) and null standard input. The command and all of
    /// its children inherit the sandbox.
    ///
    /// ```
    /// use tart_tools::sandbox::Policy;
    /// # fn main() -> anyhow::Result<()> {
    /// let policy = Policy::new(std::env::temp_dir())?.exclude_git();
    /// let status = policy.command("/usr/bin/true").status()?;
    /// assert!(status.success());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn command<S: AsRef<OsStr>>(&self, program: S) -> Command {
        let compiled = self.compile();
        let mut cmd = Command::new(SANDBOX_EXEC);
        cmd.arg("-p").arg(compiled.text);
        for (name, value) in &compiled.params {
            let mut arg = OsString::from(format!("-D{name}="));
            arg.push(value);
            cmd.arg(arg);
        }
        cmd.arg("--").arg(program.as_ref());

        // Clear the environment
        cmd.env_clear().env("PATH", SANDBOXED_PATH).stdin(Stdio::null());
        if let Some(temp) = &self.temp {
            cmd.env("TMPDIR", temp);
        }
        cmd
    }

    /// The writable roots, in grant order.
    #[must_use]
    #[inline]
    pub fn writable_roots(&self) -> &[PathBuf] {
        &self.writable
    }

    /// Render the base policy, the generated rules, and the platform defaults into a
    /// final profile.
    fn compile(&self) -> CompiledPolicy {
        let mut params = Vec::new();
        let mut read_rules = Vec::new();
        let mut write_rules = Vec::new();
        let mut deny_rules = Vec::new();

        for (i, root) in self.writable.iter().enumerate() {
            let name = format!("WRITABLE_ROOT_{i}");
            params.push((name.clone(), root.clone().into_os_string()));
            // Unguarded so excluded subpaths stay readable, just not writable.
            read_rules.push(format!(r#"(allow file-read* (subpath (param "{name}")))"#));

            let excluded = self.resolve_exclusions(root);
            if excluded.is_empty() {
                write_rules.push(format!(r#"(allow file-write* (subpath (param "{name}")))"#));
            } else {
                let mut guards = vec![format!(r#"(subpath (param "{name}"))"#)];
                for (j, excluded) in excluded.iter().enumerate() {
                    let excluded_name = format!("{name}_EXCLUDED_{j}");
                    params.push((excluded_name.clone(), excluded.clone().into_os_string()));
                    // Denies always overwrite approvals if conflicting
                    deny_rules.push(format!(
                        r#"(deny file-write* (literal (param "{excluded_name}")) (subpath (param "{excluded_name}")))"#
                    ));
                    guards.push(format!(r#"(require-not (literal (param "{excluded_name}")))"#));
                    guards.push(format!(r#"(require-not (subpath (param "{excluded_name}")))"#));
                }
                write_rules.push(format!(
                    "(allow file-write*\n  (require-all\n    {}))",
                    guards.join("\n    ")
                ));
            }
        }

        for (i, root) in self.read_only.iter().enumerate() {
            let name = format!("READABLE_ROOT_{i}");
            params.push((name.clone(), root.clone().into_os_string()));
            read_rules.push(format!(r#"(allow file-read* (subpath (param "{name}")))"#));
        }

        // Deny raw ttys to prevent sidechannel attacks
        deny_rules.push(
            r#"(deny file-read-data file-write-data file-ioctl (regex #"^/dev/ttys[0-9]+$"))"#
                .to_owned(),
        );
        deny_rules.push(r#"(deny file-read-data file-write-data (literal "/dev/tty"))"#.to_owned());

        let mut text = String::from(BASE_POLICY);
        for section in [&read_rules, &write_rules, &deny_rules] {
            if !section.is_empty() {
                text.push('\n');
                text.push_str(&section.join("\n"));
            }
        }
        text.push('\n');
        text.push_str(PLATFORM_DEFAULTS);

        CompiledPolicy { text, params }
    }

    /// Resolve the recorded exclusion specs against `root`, canonicalizing each if possible.
    fn resolve_exclusions(&self, root: &Path) -> Vec<PathBuf> {
        let mut excluded = Vec::new();
        for spec in &self.excluded {
            let candidate = root.join(spec);
            let resolved = match std::fs::canonicalize(&candidate) {
                Ok(resolved) if resolved != root && resolved.starts_with(root) => resolved,
                _ => candidate.clone(),
            };
            for path in [candidate, resolved] {
                if path != *root && !excluded.contains(&path) {
                    excluded.push(path);
                }
            }
        }
        excluded
    }
}

/// Canonicalize a granted root.
///
/// # Errors
///
/// Returns an error naming `path` when it cannot be canonicalized, most
/// commonly because it does not exist. Canonical paths are required.
fn canonicalize_root(path: &Path) -> Result<PathBuf> {
    std::fs::canonicalize(path)
        .with_context(|| format!("failed to canonicalize sandbox root: {}", path.display()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use std::ffi::OsStr;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Create a unique directory removed when the guard drops, so parallel tests don't
    /// share state.
    struct ScratchDir(PathBuf);

    impl ScratchDir {
        fn new(tag: &str) -> Self {
            Self::under(&std::env::temp_dir(), tag)
        }

        /// Create the guard under `base`, outside the temp tree when a root
        /// must not be writable by default.
        fn under(base: &Path, tag: &str) -> Self {
            // counter prevents race conditions
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            Self::at(base.join(format!("tart-sandbox-{tag}-{}-{n}", std::process::id())))
        }

        /// Creates the guard around an explicit location.
        fn at(path: PathBuf) -> Self {
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for ScratchDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The rendered profile opens with the base policy and ends with the sbpl defaults.
    #[test]
    fn render_embeds_base_policy_and_platform_defaults() {
        let dir = ScratchDir::new("render");
        let rendered = Policy::new(dir.path()).unwrap().render();
        assert!(rendered.starts_with("(version 1)"));
        assert!(rendered.contains("(deny default)"));
        assert!(rendered.contains(r#"(subpath "/opt/homebrew/lib")"#));
    }

    /// A writable root without exclusions gets one plain read allow and one plain write.
    #[test]
    fn writable_root_emits_unguarded_read_and_write_allows() {
        let dir = ScratchDir::new("writable");
        let rendered = Policy::new(dir.path()).unwrap().render();
        assert!(rendered.contains(r#"(allow file-read* (subpath (param "WRITABLE_ROOT_0")))"#));
        assert!(rendered.contains(r#"(allow file-write* (subpath (param "WRITABLE_ROOT_0")))"#));
    }

    #[test]
    fn excluded_subpath_stays_readable_but_not_writable() {
        let dir = ScratchDir::new("excluded");
        // `.git` is deliberately not created: the exclusion must still bind.
        let policy = Policy::new(dir.path()).unwrap().exclude_git();
        let rendered = policy.render();
        assert!(rendered.contains("(allow file-write*\n  (require-all"));
        assert!(
            rendered.contains(r#"(require-not (literal (param "WRITABLE_ROOT_0_EXCLUDED_0")))"#)
        );
        assert!(
            rendered.contains(r#"(require-not (subpath (param "WRITABLE_ROOT_0_EXCLUDED_0")))"#)
        );
        assert!(
            rendered.contains(
                r#"(deny file-write* (literal (param "WRITABLE_ROOT_0_EXCLUDED_0")) (subpath (param "WRITABLE_ROOT_0_EXCLUDED_0")))"#
            )
        );
        // The read allow for the same root carries no guard.
        assert!(rendered.contains(r#"(allow file-read* (subpath (param "WRITABLE_ROOT_0")))"#));

        // Lexical fallback: the binding names <root>/.git before it exists.
        let expected = OsString::from(format!(
            "-DWRITABLE_ROOT_0_EXCLUDED_0={}",
            std::fs::canonicalize(dir.path()).unwrap().join(".git").display()
        ));
        let cmd = policy.command("sh");
        assert!(cmd.get_args().any(|arg| arg == expected.as_os_str()));
    }

    /// The profile denies raw terminal devices.
    #[test]
    fn render_denies_terminal_devices() {
        let dir = ScratchDir::new("ttys");
        let rendered = Policy::new(dir.path()).unwrap().render();
        assert!(rendered.contains(
            r#"(deny file-read-data file-write-data file-ioctl (regex #"^/dev/ttys[0-9]+$"))"#
        ));
        assert!(rendered.contains(r#"(deny file-read-data file-write-data (literal "/dev/tty"))"#));
    }

    /// A read-only root gets a read allow and no write rule of its own.
    #[test]
    fn read_only_root_emits_read_allow_only() {
        let writable = ScratchDir::new("ro-w");
        let home = PathBuf::from(std::env::var_os("HOME").unwrap());
        let readable = ScratchDir::under(&home, "ro-r");
        let rendered = Policy::new(writable.path())
            .unwrap()
            .add_read_only_root(readable.path())
            .unwrap()
            .render();
        assert!(rendered.contains(r#"(allow file-read* (subpath (param "READABLE_ROOT_0")))"#));
        assert!(!rendered.contains(r#"file-write* (subpath (param "READABLE_ROOT_0")"#));
    }

    /// Roots are canonicalized, so a symlinked root grants the real path.
    #[test]
    fn roots_are_canonicalized_through_symlinks() {
        let real = ScratchDir::new("real");
        let holder = ScratchDir::new("holder");
        let link = holder.path().join("link");
        std::os::unix::fs::symlink(real.path(), &link).unwrap();
        let policy = Policy::new(&link).unwrap();
        let canonical = std::fs::canonicalize(real.path()).unwrap();
        assert_eq!(policy.writable_roots()[0], canonical);
        let binding = OsString::from(format!("-DWRITABLE_ROOT_0={}", canonical.display()));
        let cmd = policy.command("sh");
        assert!(cmd.get_args().any(|arg| arg == binding.as_os_str()));
    }

    /// Duplicate roots collapse, including the auto-added temp root.
    #[test]
    fn duplicate_roots_and_temp_dir_deduplicate() {
        let dir = ScratchDir::new("dedup");
        let policy = Policy::new(dir.path())
            .unwrap()
            .add_writable_root(dir.path())
            .unwrap();
        // The scratch root plus the temp root, nothing more.
        assert_eq!(policy.writable_roots().len(), 2);
        assert!(!policy.render().contains("WRITABLE_ROOT_2"));

        let temp = Policy::new(std::env::temp_dir()).unwrap();
        assert_eq!(temp.writable_roots().len(), 1);
        assert_eq!(
            temp.writable_roots()[0],
            std::fs::canonicalize(std::env::temp_dir()).unwrap()
        );
    }

    /// Roots that do not exist are rejected at construction, naming the path.
    #[test]
    fn nonexistent_roots_are_rejected_at_construction() {
        let missing = std::env::temp_dir().join("tart-sandbox-does-not-exist");
        let err = Policy::new(&missing).unwrap_err().to_string();
        assert!(err.contains("tart-sandbox-does-not-exist"));
        let dir = ScratchDir::new("missing");
        assert!(
            Policy::new(dir.path())
                .unwrap()
                .add_writable_root(&missing)
                .is_err()
        );
        assert!(
            Policy::new(dir.path())
                .unwrap()
                .add_read_only_root(&missing)
                .is_err()
        );
    }

    /// Empty, absolute, escaping, and root-identical exclusions are rejected.
    #[test]
    fn bad_exclusions_are_rejected() {
        let dir = ScratchDir::new("bad-exclusion");
        let policy = || Policy::new(dir.path()).unwrap();
        for (path, needle) in [
            ("", "empty"),
            ("/", "absolute"),
            ("../elsewhere", "escapes"),
            ("a/../../x", "escapes"),
            (".", "writable root itself"),
        ] {
            let err = policy().exclude(path).unwrap_err().to_string();
            assert!(err.contains(needle), "{path}: {err}");
        }
    }

    /// A read-only root overlapping a writable root in either direction is rejected
    /// instead of silently staying writable.
    #[test]
    fn overlapping_roots_are_rejected_in_both_grant_orders() {
        let outer = ScratchDir::new("overlap");
        let inner = ScratchDir::at(outer.path().join("inner"));

        let err = Policy::new(outer.path())
            .unwrap()
            .add_read_only_root(inner.path())
            .unwrap_err()
            .to_string();
        assert!(err.contains("overlaps"));

        let err = Policy::new(inner.path())
            .unwrap()
            .add_read_only_root(outer.path())
            .unwrap_err()
            .to_string();
        assert!(err.contains("overlaps"));

        // The reverse grant order must be rejected too: a writable root over
        // an existing read-only grant. System paths stand in for the read-only
        // root because scratch dirs live under the temp writable root.
        let dir = ScratchDir::new("overlap-rev");
        let err = Policy::new(dir.path())
            .unwrap()
            .add_read_only_root("/usr/local")
            .unwrap()
            .add_writable_root("/usr/local/bin")
            .unwrap_err()
            .to_string();
        assert!(err.contains("overlaps"));
    }

    /// A symlinked exclusion binds both its name and its resolved target, so the deny
    /// survives unlinking and recreating the name.
    #[test]
    fn symlinked_exclusion_denies_name_and_target() {
        let dir = ScratchDir::new("symlinked");
        std::fs::create_dir_all(dir.path().join("target")).unwrap();
        std::os::unix::fs::symlink("target", dir.path().join(".git")).unwrap();
        let policy = Policy::new(dir.path()).unwrap().exclude_git();
        let canonical = std::fs::canonicalize(dir.path()).unwrap();
        let args: Vec<String> = policy
            .command("sh")
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        let name = format!(
            "-DWRITABLE_ROOT_0_EXCLUDED_0={}",
            canonical.join(".git").display()
        );
        let target = format!(
            "-DWRITABLE_ROOT_0_EXCLUDED_1={}",
            canonical.join("target").display()
        );
        assert!(args.contains(&name));
        assert!(args.contains(&target));
    }

    /// Exclusions bind to roots added before or after them, exactly once.
    #[test]
    fn exclusions_apply_to_roots_added_in_any_order() {
        let a = ScratchDir::new("order-a");
        let b = ScratchDir::new("order-b");
        let git_exclusions = |policy: &Policy| -> Vec<String> {
            policy
                .command("sh")
                .get_args()
                .map(|arg| arg.to_string_lossy().into_owned())
                .filter(|arg| arg.starts_with("-D") && arg.ends_with("/.git"))
                .collect()
        };
        let expected_a = std::fs::canonicalize(a.path())
            .unwrap()
            .join(".git")
            .display()
            .to_string();
        let expected_b = std::fs::canonicalize(b.path())
            .unwrap()
            .join(".git")
            .display()
            .to_string();

        let before = Policy::new(a.path())
            .unwrap()
            .exclude_git()
            .add_writable_root(b.path())
            .unwrap();
        let exclusions = git_exclusions(&before);
        assert_eq!(exclusions.len(), before.writable_roots().len());
        assert!(exclusions.iter().any(|ex| ex.ends_with(&expected_a)));
        assert!(exclusions.iter().any(|ex| ex.ends_with(&expected_b)));

        let after = Policy::new(b.path())
            .unwrap()
            .add_writable_root(a.path())
            .unwrap()
            .exclude_git();
        let exclusions = git_exclusions(&after);
        assert_eq!(exclusions.len(), after.writable_roots().len());
        assert!(exclusions.iter().any(|ex| ex.ends_with(&expected_a)));
        assert!(exclusions.iter().any(|ex| ex.ends_with(&expected_b)));

        // Re-excluding is a no-op.
        let again = before.exclude_git();
        assert_eq!(git_exclusions(&again).len(), again.writable_roots().len());
    }

    /// The built command is `sandbox-exec -p <policy> -D... -- program`.
    #[test]
    fn command_prefixes_sandbox_exec_policy_params_and_separator() {
        let dir = ScratchDir::new("argv");
        let policy = Policy::new(dir.path()).unwrap().exclude_git();
        let cmd = policy.command("sh");
        assert_eq!(cmd.get_program(), OsStr::new("/usr/bin/sandbox-exec"));
        let args: Vec<&OsStr> = cmd.get_args().collect();
        assert_eq!(args[0], OsStr::new("-p"));
        assert_eq!(args[1], OsStr::new(policy.render().as_str()));
        // One binding per param, in compile order: each root followed by its
        // exclusions.
        let bindings: Vec<&OsStr> = args
            .iter()
            .copied()
            .filter(|arg| arg.to_string_lossy().starts_with("-D"))
            .collect();
        assert_eq!(bindings.len(), 2 * policy.writable_roots().len());
        assert!(bindings[0].to_string_lossy().starts_with("-DWRITABLE_ROOT_0="));
        assert!(
            bindings[1]
                .to_string_lossy()
                .starts_with("-DWRITABLE_ROOT_0_EXCLUDED_0=")
        );
        assert_eq!(args[args.len() - 2], OsStr::new("--"));
        assert_eq!(args[args.len() - 1], OsStr::new("sh"));
    }

    /// The child environment carries only the seeded variables.
    #[test]
    fn command_clears_the_environment() {
        let dir = ScratchDir::new("env");
        let policy = Policy::new(dir.path()).unwrap();
        let cmd = policy.command("sh");
        let vars: Vec<(&OsStr, &OsStr)> = cmd
            .get_envs()
            .filter_map(|(key, value)| value.map(|value| (key, value)))
            .collect();
        assert_eq!(vars.len(), 1 + usize::from(policy.temp.is_some()));
        assert!(vars.contains(&(OsStr::new("PATH"), OsStr::new(SANDBOXED_PATH))));
        if let Some(temp) = &policy.temp {
            assert!(vars.contains(&(OsStr::new("TMPDIR"), temp.as_os_str())));
        }
    }

    /// Callers extend the command with plain std methods, after `--`.
    #[test]
    fn command_composes_with_std_command_methods() {
        let dir = ScratchDir::new("compose");
        let policy = Policy::new(dir.path()).unwrap();
        let mut cmd = policy.command("sh");
        cmd.arg("-c").arg("echo hi");
        let args: Vec<&OsStr> = cmd.get_args().collect();
        let separator = args.iter().position(|arg| *arg == OsStr::new("--")).unwrap();
        assert_eq!(&args[separator + 1..], &["sh", "-c", "echo hi"]);
    }

    /// Writing inside the granted root succeeds. Paths in these live tests
    /// contain no spaces, so sh string interpolation is safe.
    #[test]
    fn write_inside_writable_root_succeeds() {
        let dir = ScratchDir::new("live-write-ok");
        let policy = Policy::new(dir.path()).unwrap();
        let file = dir.path().join("file");
        let out = policy
            .command("/bin/sh")
            .arg("-c")
            .arg(format!("echo hi > {}", file.display()))
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "hi\n");
    }

    /// Writing outside every granted root is denied by the sandbox.
    #[test]
    fn write_outside_granted_roots_is_denied() {
        let dir = ScratchDir::new("live-write-denied");
        let policy = Policy::new(dir.path()).unwrap();
        let home = PathBuf::from(std::env::var_os("HOME").unwrap());
        let target = home.join(format!(".tart-sandbox-deny-{}", std::process::id()));
        let out = policy
            .command("/bin/sh")
            .arg("-c")
            .arg(format!("echo x > {}", target.display()))
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&target);
        assert!(!out.status.success());
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("Operation not permitted"),
            "stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// An excluded `.git` stays readable but rejects writes.
    #[test]
    fn excluded_git_is_read_only_but_readable() {
        let home = PathBuf::from(std::env::var_os("HOME").unwrap());
        let dir = ScratchDir::under(&home, "git");
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        let head = dir.path().join(".git/HEAD");
        std::fs::write(&head, "ref: refs/heads/main\n").unwrap();
        let policy = Policy::new(dir.path()).unwrap().exclude_git();

        let read = policy
            .command("/bin/sh")
            .arg("-c")
            .arg(format!("cat {}", head.display()))
            .output()
            .unwrap();
        assert!(
            read.status.success(),
            "stderr: {}",
            String::from_utf8_lossy(&read.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&read.stdout), "ref: refs/heads/main\n");

        let write = policy
            .command("/bin/sh")
            .arg("-c")
            .arg(format!("echo x > {}", dir.path().join(".git/config").display()))
            .output()
            .unwrap();
        assert!(!write.status.success());
        assert!(
            String::from_utf8_lossy(&write.stderr).contains("Operation not permitted"),
            "stderr: {}",
            String::from_utf8_lossy(&write.stderr)
        );
    }

    /// A read-only root outside the temp tree reads but rejects writes.
    #[test]
    fn read_only_root_denies_writes() {
        let writable = ScratchDir::new("live-ro-w");
        // Under HOME, outside the temp writable root, so only the read-only
        // grant makes it accessible.
        let home = PathBuf::from(std::env::var_os("HOME").unwrap());
        let readonly = ScratchDir::under(&home, "ro-live");
        let data = readonly.path().join("data");
        std::fs::write(&data, "contents\n").unwrap();
        let policy = Policy::new(writable.path())
            .unwrap()
            .add_read_only_root(readonly.path())
            .unwrap();

        let denied = policy
            .command("/bin/sh")
            .arg("-c")
            .arg(format!("echo x > {}", readonly.path().join("x").display()))
            .output()
            .unwrap();
        assert!(!denied.status.success());
        assert!(
            String::from_utf8_lossy(&denied.stderr).contains("Operation not permitted"),
            "stderr: {}",
            String::from_utf8_lossy(&denied.stderr)
        );

        let read = policy
            .command("/bin/sh")
            .arg("-c")
            .arg(format!("cat {}", data.display()))
            .output()
            .unwrap();
        assert!(
            read.status.success(),
            "stderr: {}",
            String::from_utf8_lossy(&read.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&read.stdout), "contents\n");

        let allowed = policy
            .command("/bin/sh")
            .arg("-c")
            .arg(format!("echo x > {}", writable.path().join("x").display()))
            .output()
            .unwrap();
        assert!(
            allowed.status.success(),
            "stderr: {}",
            String::from_utf8_lossy(&allowed.stderr)
        );
    }
}
