/// Progress from one background generation.
///
/// This struct is a compatibility shim to separate `async-openai` generation types from
/// terminal frontend. As a side benefit, it gives us more control over how we store and
/// track history/context.
#[derive(Debug)]
#[non_exhaustive]
pub enum Progress {
    /// A fragment of the model's reasoning.
    Thinking(String),
    /// A fragment of the final answer.
    Answer(String),
    /// A command the model asked to run.
    Command(String),
    /// The combined output of a finished command.
    CommandOutput(String),
    /// The assembled answer at the end of the stream, if any arrived.
    Done {
        /// The full answer text, unless the model produced `None`.
        message: Option<String>,
    },
    /// The request or stream failed.
    Failed(String),
}
