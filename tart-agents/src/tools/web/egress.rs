//! Destination admission for `fetch`
//!
//! Every name resolves through [`PublicDns`], which hands the connector only addresses
//! [`is_global`] admits for security.

use std::net::{IpAddr, SocketAddr};

use anyhow::{Context as _, ensure};
use iprfc::RFC6890;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use url::{Host, Url};

/// Admit `url` as an http(s) URL whose host is on the public web.
#[allow(
    clippy::case_sensitive_file_extension_comparisons,
    reason = "these are domain suffixes, and the parser lowercased the host"
)]
pub(super) fn check_destination(url: &Url) -> anyhow::Result<()> {
    ensure!(
        matches!(url.scheme(), "http" | "https"),
        "fetch: only http and https are supported, not {}",
        url.scheme()
    );
    match url.host() {
        None => anyhow::bail!("fetch: url has no host"),
        Some(Host::Domain(name)) => {
            let name = name.strip_suffix('.').unwrap_or(name);
            let private = name == "localhost"
                || name.ends_with(".localhost")
                || name.ends_with(".local")
                || name.ends_with(".internal")
                // A zone ID names a link-local interface address.
                || name.contains('%');
            ensure!(!private, "fetch: refusing non-public host {name}");
        }
        Some(Host::Ipv4(ip)) => {
            ensure!(is_global(ip.into()), "fetch: refusing non-public address {ip}");
        }
        Some(Host::Ipv6(ip)) => {
            ensure!(is_global(ip.into()), "fetch: refusing non-public address {ip}");
        }
    }
    Ok(())
}

/// Whether `ip` is on the public web: not multicast, and not a row of the IANA
/// special-purpose registries as [`RFC6890`] generates them.
///
/// `std::net::IpAddr::is_global` replaces this once stable.
#[must_use]
pub(super) fn is_global(ip: IpAddr) -> bool {
    let ip = match ip {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(ip, IpAddr::V4),
        IpAddr::V4(_) => ip,
    };
    !ip.is_multicast() && !RFC6890.contains(&ip)
}

/// The shared client's resolver: one lookup per request, only [`is_global`]
/// addresses reach the connector.
pub(super) struct PublicDns;

impl Resolve for PublicDns {
    fn resolve(&self, name: Name) -> Resolving {
        // The root dot names the same host; dropped so messages match the name rules.
        let host = name.as_str().trim_end_matches('.').to_owned();
        Box::pin(async move {
            tokio::net::lookup_host((host.as_str(), 0))
                .await
                .with_context(|| format!("fetch: could not resolve {host}"))
                .and_then(|addrs| admitted(&host, addrs.map(|addr| addr.ip()).collect()))
                .map_err(Into::into)
        })
    }
}

/// `answers` as the connector may dial them: every one admitted, each with port
/// zero so the URL's own port applies. One disallowed answer refuses the set.
fn admitted(host: &str, answers: Vec<IpAddr>) -> anyhow::Result<Addrs> {
    if let Some(ip) = answers.iter().find(|ip| !is_global(**ip)) {
        anyhow::bail!("fetch: refusing non-public address {ip} answered for {host}");
    }
    ensure!(!answers.is_empty(), "fetch: {host} resolved to no addresses");
    Ok(Box::new(answers.into_iter().map(|ip| SocketAddr::new(ip, 0))) as Addrs)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test assertions")]

    use std::net::Ipv4Addr;

    use super::*;

    fn v4(a: u8, b: u8, c: u8, d: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(a, b, c, d))
    }

    fn v6(text: &str) -> IpAddr {
        text.parse().unwrap()
    }

    /// `run_fetch`'s own step: parse with the client's parser, then judge.
    fn admitted_url(input: &str) -> Result<Url, String> {
        Url::parse(input.trim())
            .with_context(|| format!("cannot parse {input:?}"))
            .and_then(|url| check_destination(&url).map(|()| url))
            .map_err(|error| error.to_string())
    }

    /// Every parser differential from the audit, one form per class: the
    /// client's parser folds each into a typed host the same rule refuses.
    #[test]
    fn check_destination_admits_public_web_addresses_and_refuses_the_differentials() {
        for url in [
            "HTTP://User:Pw@Example.COM:8080/x?q",
            "http://93.184.216.34:8080/page",
            "https://[2606:4700:4700::1111]/",
            "https://example.com./",
        ] {
            assert!(admitted_url(url).is_ok(), "{url}");
        }
        for url in [
            "http://127.0.0.1\\@example.com/", // a backslash ends the authority
            "http://169.254.169.254\\@example.com/x", // ... hiding the metadata address
            "http://0177.0.0.1/",              // octal IPv4
            "http://127.0.0.1./",              // trailing root dot
            "http://１２７。０。０。１/",      // fullwidth digits, ideographic dots
            "http://ｌocalhost/",              // IDNA letters
            "http://[::ffff:127.0.0.1]/",      // IPv4-mapped
            "http://[64:ff9b::127.0.0.1]/",    // NAT64
            "https://[fe80::1%25en0]/",        // zone ID
            "https://Printer.Local./",         // a name, on the normalized domain
        ] {
            assert!(admitted_url(url).is_err(), "expected refusal: {url}");
        }
        for (url, refusal) in [
            ("https://LOCALHOST/x", "fetch: refusing non-public host localhost"),
            (
                "ftp://example.com/x",
                "fetch: only http and https are supported, not ftp",
            ),
        ] {
            assert_eq!(admitted_url(url).unwrap_err(), refusal, "{url}");
        }
    }

    /// Every registry row (each net's first address), multicast, and a mapped
    /// private address are refused; the space between ranges and the delisted
    /// deprecated ranges are admitted.
    #[test]
    fn is_global_is_the_registry_plus_multicast() {
        for net in RFC6890.ip_nets() {
            assert!(!is_global(net.network()), "{net}");
        }
        for ip in [v4(224, 0, 0, 1), v6("ff02::1"), v6("::ffff:10.0.0.1")] {
            assert!(!is_global(ip), "expected refusal: {ip}");
        }
        for ip in [
            v4(100, 128, 0, 1),
            v4(223, 255, 255, 255),
            v6("2606:4700:4700::1111"),
            v6("fe7f::1"),
            v6("fec0::1"),
            v6("::127.0.0.1"),
            v6("::ffff:8.8.8.8"),
        ] {
            assert!(is_global(ip), "expected admission: {ip}");
        }
    }

    /// The connector receives exactly the admitted answers, with port zero; one
    /// disallowed answer, or none at all, refuses the whole set.
    #[test]
    fn the_connector_receives_only_admitted_answers() {
        let addrs: Vec<_> = admitted("example.com", vec![v4(93, 184, 216, 34)])
            .unwrap()
            .collect();
        assert_eq!(addrs, [SocketAddr::new(v4(93, 184, 216, 34), 0)]);
        assert_eq!(
            admitted("rebind.example", vec![v4(8, 8, 8, 8), v4(10, 0, 0, 5)])
                .err()
                .unwrap()
                .to_string(),
            "fetch: refusing non-public address 10.0.0.5 answered for rebind.example"
        );
        assert_eq!(
            admitted("empty.example", Vec::new()).err().unwrap().to_string(),
            "fetch: empty.example resolved to no addresses"
        );
    }
}
