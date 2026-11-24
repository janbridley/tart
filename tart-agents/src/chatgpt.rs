//! Local Sign in with `ChatGPT` credentials.

use std::fs::File;
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, bail, ensure};
use openidconnect::core::{
    CoreAuthenticationFlow, CoreClient, CoreJsonWebKeySet, CoreProviderMetadata,
};
use openidconnect::{
    AuthType, AuthorizationCode, ClientId, CsrfToken, HttpClientError, HttpRequest, HttpResponse,
    IssuerUrl, Nonce, OAuth2TokenResponse, PkceCodeChallenge, PkceCodeVerifier, RedirectUrl,
    RefreshToken, RequestTokenError, Scope, SyncHttpClient, TokenResponse, TokenUrl,
};
use reqwest::blocking::{Client, ClientBuilder};
use security_framework::passwords::{
    delete_generic_password, get_generic_password, set_generic_password,
};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

pub(crate) const RESOURCE: &str = "https://api.openai.com/v1";
const ISSUER: &str = "https://auth.openai.com";
const TOKEN_URL: &str = "https://auth.openai.com/api/accounts/oauth/token";
const DYNAMIC_CLIENT: &str = "dynamic_agent_client";
pub(crate) const RECONNECT: &str = "Run tart login chatgpt to connect your account";
const RESET: &str =
    "ChatGPT rejected the registration. Run tart login chatgpt --reset to register again.";
const SESSION_LOCK: &str = "chatgpt.lock";
const ITEM_NOT_FOUND: i32 = -25300;
const SERVICE: &str = "tart-chatgpt";
const SCOPE: &str = "chatgpt.tokens.use.direct";
const CALLBACK: &str = "/auth/callback";

#[derive(Deserialize, Serialize)]
struct Registration {
    host: String,
    // Older records kept the issued ID here even after verification.
    #[serde(alias = "client_id")]
    pending_client_id: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct Session {
    client_id: String,
    subject: String,
    access: Option<String>,
    refresh: Option<String>,
    expires: i64,
}

fn load_keychain<T: serde::de::DeserializeOwned>(account: &str) -> anyhow::Result<Option<T>> {
    match get_generic_password(SERVICE, account) {
        Ok(bytes) => Ok(Some(
            serde_json::from_slice(&bytes).context("Invalid Tart Keychain entry")?,
        )),
        Err(error) if error.code() == ITEM_NOT_FOUND => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn save_keychain(account: &str, value: &impl Serialize) -> anyhow::Result<()> {
    Ok(set_generic_password(
        SERVICE,
        account,
        &serde_json::to_vec(value)?,
    )?)
}

fn lock(name: &str) -> anyhow::Result<File> {
    let home = std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .context("HOME is required for ChatGPT sign-in")?;
    let dir = std::path::PathBuf::from(home).join(".config/tart");
    std::fs::create_dir_all(&dir)?;
    let file = File::options()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(dir.join(name))?;
    file.lock()?;
    Ok(file)
}

struct OAuthHttp(Client);

impl SyncHttpClient for OAuthHttp {
    type Error = HttpClientError<reqwest::Error>;

    fn call(&self, request: HttpRequest) -> Result<HttpResponse, Self::Error> {
        let response = self
            .0
            .execute(request.try_into().map_err(Box::new)?)
            .map_err(Box::new)?;
        let status = response.status();
        if status.is_server_error() || status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            response.error_for_status_ref().map_err(Box::new)?;
        }
        let mut result = HttpResponse::new(Vec::new());
        *result.status_mut() = status;
        *result.headers_mut() = response.headers().clone();
        *result.body_mut() = response.bytes().map_err(Box::new)?.to_vec();
        Ok(result)
    }
}

impl TryFrom<ClientBuilder> for OAuthHttp {
    type Error = reqwest::Error;

    fn try_from(builder: ClientBuilder) -> Result<Self, Self::Error> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        builder
            .https_only(true)
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(20))
            .build()
            .map(Self)
    }
}

fn callback(
    server: &tiny_http::Server,
    state: &CsrfToken,
) -> anyhow::Result<(AuthorizationCode, Option<String>)> {
    // Wait for the browser's redirect; Ctrl-C abandons the sign-in. Stray
    // requests (favicons, probes) are answered and ignored.
    for request in server.incoming_requests() {
        let Some(result) = parse_callback(request.method(), request.url(), state) else {
            let _ = request.respond(tiny_http::Response::empty(400));
            continue;
        };
        let message = if result.is_ok() {
            "Return to Tart to finish signing in."
        } else {
            "ChatGPT sign-in did not complete. Return to Tart."
        };
        let _ = request.respond(tiny_http::Response::from_string(message));
        return result;
    }
    anyhow::bail!("ChatGPT sign-in stopped before the browser returned")
}

fn parse_callback(
    method: &tiny_http::Method,
    target: &str,
    state: &CsrfToken,
) -> Option<anyhow::Result<(AuthorizationCode, Option<String>)>> {
    let (path, query) = target.split_once('?')?;
    (method == &tiny_http::Method::Get && path == CALLBACK).then_some(())?;
    let params = url::form_urlencoded::parse(query.as_bytes());
    let get = |name: &str| -> anyhow::Result<Option<String>> {
        let values: Vec<_> = params.filter(|(key, _)| key == name).collect();
        match &values[..] {
            [] => Ok(None),
            [(_, value)] => Ok(Some(value.to_string())),
            _ => bail!("Duplicate ChatGPT callback parameter"),
        }
    };
    if get("state").ok().flatten().as_ref() != Some(state.secret()) {
        return None;
    }
    Some((|| {
        ensure!(get("error")?.is_none(), "ChatGPT sign-in was declined or failed");
        let code = get("code")?
            .filter(|code| !code.is_empty())
            .context("ChatGPT did not return a code")?;
        Ok((AuthorizationCode::new(code), get("client_id")?))
    })())
}

struct Authorization {
    code: AuthorizationCode,
    client_id: String,
    redirect: RedirectUrl,
    verifier: PkceCodeVerifier,
    nonce: Nonce,
}

fn authorize_in_browser(
    metadata: &CoreProviderMetadata,
    host: &str,
    saved_id: Option<&str>,
    needs_consent: bool,
) -> anyhow::Result<Authorization> {
    let server = tiny_http::Server::http("127.0.0.1:0").map_err(anyhow::Error::from_boxed)?;
    let redirect = RedirectUrl::new(format!("http://{}{CALLBACK}", server.server_addr()))?;
    let client = CoreClient::from_provider_metadata(
        metadata.clone(),
        ClientId::new(saved_id.unwrap_or(DYNAMIC_CLIENT).into()),
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
        .add_extra_param("ext_agent_host_id", host)
        .add_scopes(
            ["profile", "email", "offline_access", "resource.invoke", SCOPE]
                .map(|scope| Scope::new(scope.into())),
        );
    if saved_id.is_none() {
        authorize = authorize.add_extra_param("agent_name_hint", "Tart");
    } else if needs_consent {
        authorize = authorize.add_extra_param("prompt", "consent");
    }
    let (url, state, nonce) = authorize.url();
    let opened = Command::new("/usr/bin/open")
        .arg(url.as_str())
        .status()?
        .success();
    ensure!(opened, "Could not open the browser");
    let (code, returned_id) = callback(&server, &state)?;
    let client_id = match (saved_id, returned_id) {
        (Some(saved), Some(returned)) if saved != returned => {
            bail!("ChatGPT returned a different client ID")
        }
        (Some(saved), _) => saved.to_owned(),
        (None, returned) => returned
            .filter(|id| !id.is_empty() && id != DYNAMIC_CLIENT)
            .context("ChatGPT did not return a registered client ID")?,
    };
    Ok(Authorization {
        code,
        client_id,
        redirect,
        verifier,
        nonce,
    })
}

/// Sign in to `ChatGPT` and save the session in Keychain.
///
/// Pass `reset` to clear the connection and register again with the same host ID.
#[inline]
pub fn login(reset: bool) -> anyhow::Result<()> {
    // Serialize sign-ins without blocking inference during browser interaction.
    let _login_lock = lock("chatgpt-login.lock")?;
    let (mut registration, saved_id, needs_consent) = {
        let _lock = lock(SESSION_LOCK)?;
        let mut registration =
            load_keychain::<Registration>("registration")?.unwrap_or_else(|| Registration {
                host: format!("urn:uuid:{}", uuid::Uuid::new_v4()),
                pending_client_id: None,
            });
        if reset {
            match delete_generic_password(SERVICE, "session") {
                Ok(()) => (),
                Err(error) if error.code() == ITEM_NOT_FOUND => (),
                Err(error) => return Err(error.into()),
            }
            registration.pending_client_id = None;
        }
        let saved = load_keychain::<Session>("session")?;
        let needs_consent = saved.as_ref().is_none_or(|session| session.access.is_none());
        let saved_id = saved
            .map(|session| session.client_id)
            .or_else(|| registration.pending_client_id.clone());
        save_keychain("registration", &registration)?;
        (registration, saved_id, needs_consent)
    };
    let http = OAuthHttp::try_from(Client::builder())?;
    let metadata = CoreProviderMetadata::discover(&IssuerUrl::new(ISSUER.into())?, &http)?;
    let authorization =
        authorize_in_browser(&metadata, &registration.host, saved_id.as_deref(), needs_consent)?;
    let _lock = lock(SESSION_LOCK)?;
    if saved_id.is_none() {
        // The issued ID can be reused if the authorization code expires.
        registration.pending_client_id = Some(authorization.client_id.clone());
        save_keychain("registration", &registration)?;
    }
    let session = exchange_code(metadata, &http, authorization)?;
    if let Some(saved) = load_keychain::<Session>("session")? {
        ensure!(
            session.client_id == saved.client_id && session.subject == saved.subject,
            "ChatGPT returned a different account for the saved registration"
        );
    }
    save_keychain("session", &session)?;
    if registration.pending_client_id.take().is_some() {
        save_keychain("registration", &registration)?;
    }
    Ok(())
}

fn exchange_code(
    metadata: CoreProviderMetadata,
    http: &OAuthHttp,
    authorization: Authorization,
) -> anyhow::Result<Session> {
    let client = CoreClient::from_provider_metadata(
        metadata,
        ClientId::new(authorization.client_id.clone()),
        None,
    )
    .set_auth_type(AuthType::RequestBody)
    .set_redirect_uri(authorization.redirect);
    let tokens = client
        .exchange_code(authorization.code)?
        .set_pkce_verifier(authorization.verifier)
        .add_extra_param("resource", RESOURCE)
        .request(http)
        .map_err(|error| match error {
            RequestTokenError::ServerResponse(error)
                if error.error().as_ref() == "invalid_client" =>
            {
                anyhow::anyhow!(RESET)
            }
            _ => anyhow::anyhow!("ChatGPT code exchange failed. Run tart login chatgpt again."),
        })?;
    let claims = tokens
        .id_token()
        .context("ChatGPT did not return an ID token")?
        .claims(&client.id_token_verifier(), &authorization.nonce)?;
    ensure!(
        tokens
            .scopes()
            .is_some_and(|scopes| scopes.iter().any(|scope| scope.as_str() == SCOPE)),
        "ChatGPT plan usage was not granted"
    );
    let refresh = tokens
        .refresh_token()
        .context("ChatGPT did not return a refresh token")?;
    Ok(Session {
        client_id: authorization.client_id,
        subject: claims.subject().as_str().into(),
        access: Some(tokens.access_token().secret().clone()),
        refresh: Some(refresh.secret().clone()),
        expires: expires_at(tokens.expires_in())?,
    })
}

fn expires_at(lifetime: Option<Duration>) -> anyhow::Result<i64> {
    let seconds = lifetime.context("ChatGPT did not return token expiry")?.as_secs();
    Ok(OffsetDateTime::now_utc()
        .unix_timestamp()
        .saturating_add(i64::try_from(seconds)?))
}

pub(crate) fn access_token() -> anyhow::Result<String> {
    let _lock = lock(SESSION_LOCK)?;
    let mut session = load_keychain::<Session>("session")?.context(RECONNECT)?;
    if session.access.is_some()
        && session.expires <= OffsetDateTime::now_utc().unix_timestamp() + 60
    {
        refresh(&mut session)?;
    }
    session.access.context(RECONNECT)
}

fn clear_tokens(session: &mut Session) -> anyhow::Result<()> {
    session.access = None;
    session.refresh = None;
    save_keychain("session", session)
}

pub(crate) fn invalidate_access_token(access: &str) -> anyhow::Result<()> {
    let _lock = lock(SESSION_LOCK)?;
    if let Some(mut session) = load_keychain::<Session>("session")?
        && session.access.as_deref() == Some(access)
    {
        clear_tokens(&mut session)?;
    }
    Ok(())
}

fn refresh(session: &mut Session) -> anyhow::Result<()> {
    let client = CoreClient::new(
        ClientId::new(session.client_id.clone()),
        IssuerUrl::new(ISSUER.into())?,
        CoreJsonWebKeySet::default(),
    )
    .set_token_uri(TokenUrl::new(TOKEN_URL.into())?)
    .set_auth_type(AuthType::RequestBody);
    let refresh = RefreshToken::new(session.refresh.clone().context(RECONNECT)?);
    let result = client
        .exchange_refresh_token(&refresh)
        .add_extra_param("resource", RESOURCE)
        .request(&OAuthHttp::try_from(Client::builder())?);
    let tokens = match result {
        Ok(tokens) => tokens,
        Err(error) => {
            let transient = match error {
                RequestTokenError::Request(_) => true,
                RequestTokenError::ServerResponse(error) => match error.error().as_ref() {
                    "invalid_grant"
                    | "invalid_refresh_token"
                    | "token_expired"
                    | "refresh_token_expired"
                    | "refresh_token_invalidated"
                    | "refresh_token_reused" => {
                        clear_tokens(session)?;
                        bail!("ChatGPT session expired or was revoked. {RECONNECT}.");
                    }
                    "invalid_client" => bail!(RESET),
                    "temporarily_unavailable" | "server_error" => true,
                    _ => false,
                },
                _ => false,
            };
            if transient && session.expires > OffsetDateTime::now_utc().unix_timestamp() {
                return Ok(());
            }
            bail!("ChatGPT token refresh failed. Retry or run tart login chatgpt.");
        }
    };
    if tokens
        .scopes()
        .is_some_and(|scopes| !scopes.iter().any(|scope| scope.as_str() == SCOPE))
    {
        clear_tokens(session)?;
        bail!("ChatGPT plan usage was not granted. {RECONNECT}.");
    }
    session.expires = expires_at(tokens.expires_in())?;
    session.access = Some(tokens.access_token().secret().clone());
    if let Some(replacement) = tokens.refresh_token() {
        session.refresh = Some(replacement.secret().clone());
    }
    save_keychain("session", session)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiny_http::Method;

    #[test]
    fn callback_requires_matching_state_and_prefers_oauth_errors() {
        let state = CsrfToken::new("expected".into());
        let callback = |target| parse_callback(&Method::Get, target, &state);
        assert!(callback("/auth/callback?state=wrong&code=example").is_none());
        assert!(callback("/auth/callback?state=expected&state=expected&code=example").is_none());
        assert!(callback("/another/path?state=expected&code=example").is_none());
        assert!(matches!(
            callback("/auth/callback?state=expected&code=example&error=access_denied"),
            Some(Err(_))
        ));
        let (code, id) = callback("/auth/callback?state=expected&code=example%2Bcode&client_id=id")
            .expect("callback matches")
            .expect("callback has a code");
        assert_eq!(code.secret(), "example+code");
        assert_eq!(id.as_deref(), Some("id"));
    }

    #[test]
    fn previous_keychain_records_can_be_loaded() {
        let registration: Registration =
            serde_json::from_str(r#"{"host":"host","client_id":"id"}"#).expect("old registration");
        let session: Session = serde_json::from_str(
            r#"{"client_id":"id","subject":"sub","access":"a","refresh":"r","expires":123}"#,
        )
        .expect("old session");
        assert_eq!(registration.pending_client_id.as_deref(), Some("id"));
        assert_eq!(session.access.as_deref(), Some("a"));
        assert_eq!(session.refresh.as_deref(), Some("r"));
    }
}
