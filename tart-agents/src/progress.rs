/// Progress from one background generation.
///
/// This struct is a compatibility shim to separate `async-openai` generation types from
/// terminal frontend. As a side benefit, it gives us more control over how we store and
/// track history/context.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Progress {
    /// A user message from the record; replay only, never live generation.
    User(String),
    /// A fragment of the model's reasoning.
    Thinking(String),
    /// A fragment of the final answer.
    Answer(String),
    /// A tool invocation started; fires before execution.
    ToolStart {
        /// The call's id, pairing the start with its eventual output.
        id: String,
        /// The tool's name on the wire: one of {`bash`, `read`, `edit`, `search`,
        /// `fetch`, `spawn`, `wait`}, or `agent` for a subagent's box opener.
        name: String,
        /// The call's raw JSON arguments, exactly as the provider sent them.
        arguments: String,
    },
    /// A finished tool invocation, paired with its start by `id`.
    ToolOutput {
        id: String,
        /// The combined output, shown to the user.
        output: String,
        /// The process exit code; `None` when no process ran (spawn error).
        exit: Option<i32>,
    },
    /// One finished response's token usage, as the provider measured it.
    Usage {
        /// All input tokens, cache included.
        input: u64,
        /// The input tokens served from the prompt cache.
        cached: u64,
        /// The tokens the model generated.
        output: u64,
    },
    /// The assembled answer at the end of the stream, if any arrived.
    Done {
        /// The full answer text, unless the model produced `None`.
        message: Option<String>,
    },
    /// The request or stream failed.
    Failed(String),
    /// The front end cancelled the turn and any partial answer that arrived is recorded
    Cancelled,
    /// A transient bit of text that is *not* part of the session record.
    Note(String),
}
