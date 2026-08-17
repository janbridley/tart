mod openai;

pub trait ModelConfiguration {
    /// A valid URL through which we can reach the LLM.
    fn url(&self) -> url::Url;
    /// A model name the provider recognizes.
    fn model(&self) -> String;
    /// The api key with which to authenticate requests.
    fn api_key(&self) -> String;
}

#[cfg(test)]
mod tests {
    use super::*;
}
