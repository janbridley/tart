//! Local Sign in with `ChatGPT` credentials.

use std::fs::File;
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, ensure};
use openidconnect::core::{CoreAuthenticationFlow, CoreClient, CoreProviderMetadata};
use openidconnect::reqwest::blocking::Client;
use openidconnect::{
    AuthType, AuthorizationCode, ClientId, CsrfToken, IssuerUrl, Nonce, OAuth2TokenResponse,
    PkceCodeChallenge, RedirectUrl, RefreshToken, Scope, TokenResponse,
};
use security_framework::passwords::{get_generic_password, set_generic_password};
use serde::{Deserialize, Serialize};
use url::Url;

pub(crate) const RESOURCE: &str = "https://api.openai.com/v1";
const ISSUER: &str = "https://auth.openai.com";
const SERVICE: &str = "tart-chatgpt";
const SCOPE: &str = "chatgpt.tokens.use.direct";
const CALLBACK: &str = "/auth/callback";

#[derive(Deserialize, Serialize)]
struct Registration {
    host: String,
    client_id: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct Session {
    client_id: String,
    subject: String,
    access: String,
    refresh: String,
    expires: u64,
}

fn load<T: serde::de::DeserializeOwned>(account: &str) -> anyhow::Result<Option<T>> {
    match get_generic_password(SERVICE, account) {
        Ok(bytes) => Ok(Some(
            serde_json::from_slice(&bytes).context("Invalid Tart Keychain entry")?,
        )),
        Err(error) if error.code() == -25300 => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn save(account: &str, value: &impl Serialize) -> anyhow::Result<()> {
    Ok(set_generic_password(
        SERVICE,
        account,
        &serde_json::to_vec(value)?,
    )?)
}

fn lock() -> anyhow::Result<File> {
    let home = std::env::var_os("HOME").context("HOME is required for ChatGPT sign-in")?;
    let dir = std::path::PathBuf::from(home).join(".config/tart");
    std::fs::create_dir_all(&dir)?;
    let file = File::options()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(dir.join("chatgpt.lock"))?;
    file.lock()?;
    Ok(file)
}

fn http() -> anyhow::Result<Client> {
    Ok(Client::builder()
        .redirect(openidconnect::reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(20))
        .build()?)
}

fn metadata(http: &Client) -> anyhow::Result<CoreProviderMetadata> {
    Ok(CoreProviderMetadata::discover(
        &IssuerUrl::new(ISSUER.into())?,
        http,
    )?)
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn callback(
    server: &tiny_http::Server,
    state: &CsrfToken,
) -> anyhow::Result<(String, Option<String>)> {
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        let request = server
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))?
            .context("ChatGPT sign-in timed out")?;
        let url = Url::parse(&format!("http://127.0.0.1{}", request.url()))?;
        let get = |name| {
            url.query_pairs()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.into_owned())
        };
        if url.path() != CALLBACK
            || request.method() != &tiny_http::Method::Get
            || get("state").as_ref() != Some(state.secret())
        {
            request.respond(tiny_http::Response::empty(400))?;
            continue;
        }
        let code = get("code");
        request.respond(tiny_http::Response::from_string(
            "Return to Tart to finish signing in.",
        ))?;
        return Ok((
            code.context("ChatGPT sign-in was declined or failed")?,
            get("client_id"),
        ));
    }
}

/// Sign in to `ChatGPT` in the browser and save the session in Keychain.
#[inline]
pub fn login() -> anyhow::Result<()> {
    let _lock = lock()?;
    let mut registration = load::<Registration>("registration")?.unwrap_or_else(|| Registration {
        host: format!("urn:uuid:{}", uuid::Uuid::new_v4()),
        client_id: None,
    });
    save("registration", &registration)?;
    let http = http()?;
    let metadata = metadata(&http)?;
    let server = tiny_http::Server::http("127.0.0.1:0").map_err(anyhow::Error::from_boxed)?;
    let redirect = RedirectUrl::new(format!("http://{}{CALLBACK}", server.server_addr()))?;
    let client = CoreClient::from_provider_metadata(
        metadata.clone(),
        ClientId::new(
            registration
                .client_id
                .clone()
                .unwrap_or_else(|| "dynamic_agent_client".into()),
        ),
        None,
    )
    .set_redirect_uri(redirect.clone());
    let (challenge, verifier) = PkceCodeChallenge::new_random_sha256();
    let mut authorize = client
        .authorize_url(
            CoreAuthenticationFlow::AuthorizationCode,
            CsrfToken::new_random,
            Nonce::new_random,
        )
        .set_pkce_challenge(challenge)
        .add_extra_param("resource", RESOURCE)
        .add_extra_param("ext_agent_host_id", registration.host.clone());
    if registration.client_id.is_none() {
        authorize = authorize.add_extra_param("agent_name_hint", "Tart");
    }
    for scope in ["profile", "email", "offline_access", "resource.invoke", SCOPE] {
        authorize = authorize.add_scope(Scope::new(scope.into()));
    }
    let (url, state, nonce) = authorize.url();
    ensure!(
        Command::new("/usr/bin/open")
            .arg(url.as_str())
            .status()?
            .success(),
        "Could not open the browser"
    );
    let (code, returned_id) = callback(&server, &state)?;
    let client_id = match (&registration.client_id, returned_id) {
        (Some(saved), Some(returned)) if *saved != returned => {
            anyhow::bail!("ChatGPT returned a different client ID")
        }
        (Some(saved), _) => saved.clone(),
        (None, returned) => returned
            .filter(|id| !id.is_empty() && id != "dynamic_agent_client")
            .context("ChatGPT did not return a registered client ID")?,
    };
    registration.client_id = Some(client_id.clone());
    save("registration", &registration)?;
    let client =
        CoreClient::from_provider_metadata(metadata, ClientId::new(client_id.clone()), None)
            .set_auth_type(AuthType::RequestBody)
            .set_redirect_uri(redirect);
    let tokens = client
        .exchange_code(AuthorizationCode::new(code))?
        .set_pkce_verifier(verifier)
        .add_extra_param("resource", RESOURCE)
        .request(&http)
        .map_err(|_| {
            anyhow::anyhow!("ChatGPT code exchange failed. Run tart login chatgpt again.")
        })?;
    let claims = tokens
        .id_token()
        .context("ChatGPT did not return an ID token")?
        .claims(&client.id_token_verifier(), &nonce)?;
    ensure!(
        tokens
            .scopes()
            .is_some_and(|scopes| scopes.iter().any(|scope| scope.as_str() == SCOPE)),
        "ChatGPT plan usage was not granted"
    );
    if let Some(saved) = load::<Session>("session")? {
        ensure!(
            claims.subject().as_str() == saved.subject,
            "ChatGPT returned a different account for the saved registration"
        );
    }
    save(
        "session",
        &Session {
            client_id,
            subject: claims.subject().as_str().into(),
            access: tokens.access_token().secret().clone(),
            refresh: tokens
                .refresh_token()
                .context("ChatGPT did not return a refresh token")?
                .secret()
                .clone(),
            expires: now().saturating_add(
                tokens
                    .expires_in()
                    .context("ChatGPT did not return token expiry")?
                    .as_secs(),
            ),
        },
    )
}

pub(crate) fn access_token() -> anyhow::Result<String> {
    let _lock = lock()?;
    let mut session =
        load::<Session>("session")?.context("Run tart login chatgpt to connect your account")?;
    if session.expires <= now().saturating_add(60) {
        refresh(&mut session)?;
    }
    Ok(session.access)
}

fn refresh(session: &mut Session) -> anyhow::Result<()> {
    let http = http()?;
    let client = CoreClient::from_provider_metadata(
        metadata(&http)?,
        ClientId::new(session.client_id.clone()),
        None,
    )
    .set_auth_type(AuthType::RequestBody);
    let refresh = RefreshToken::new(session.refresh.clone());
    let result = client
        .exchange_refresh_token(&refresh)?
        .add_extra_param("resource", RESOURCE)
        .request(&http);
    let tokens = match result {
        Ok(tokens) => tokens,
        Err(_) if session.expires > now() => return Ok(()),
        Err(_) => anyhow::bail!("ChatGPT token refresh failed. Retry or run tart login chatgpt."),
    };
    ensure!(
        tokens
            .scopes()
            .is_none_or(|scopes| scopes.iter().any(|scope| scope.as_str() == SCOPE)),
        "ChatGPT plan usage was not granted"
    );
    session.access.clone_from(tokens.access_token().secret());
    if let Some(replacement) = tokens.refresh_token() {
        session.refresh.clone_from(replacement.secret());
    }
    session.expires = now().saturating_add(
        tokens
            .expires_in()
            .context("ChatGPT did not return token expiry")?
            .as_secs(),
    );
    save("session", session)
}
