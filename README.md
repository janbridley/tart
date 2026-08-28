# *tart*

*tart*: like a pi you can see inside! *tart* is a lightweight, auditable agent harness
implemented in less than 10K lines of code. Though minimal, *tart* implements a variety
of nice-to-have features including native tmux scrollback, OS-level sandboxing,
`@filename-mentions`, inline bash commands via `!script.sh`, a read-only plan mode, and
subagents. Tool calls are sandboxed by default, so tools can run without approval.

## Architecture

### Sandboxing

Sandboxing is *always* enabled for tart, and there is no way for the model to request
outside access or edit files outside the sandbox. As a result, all bash commands and
filesystem tools run without approval. In practice, this means that the model is given
unrestricted access to both the working directory, `/tmp`, and `$TMPDIR`, so plan
accordingly. Web tools are an exception to this, as they are permitted web access but
are unable to read or write to disk.

We choose `sandbox-exec` as a lightweight sandbox option, as it effectively balances
safety with memory usage and complexity. While running the agent in a container seems
beneficial, the real safety bottleneck is syncing local data with the agent's copies.
Manual approvals make this step safe in theory, but agentic workloads are capable of
generating far more requests for approval of complex commands then can reasonably be
evaluated. *Sandboxing is helpful, but you should always use version control and backup
tools if your working directory contains data that is not stored elsewhere!*

### Tools

*tart* provides five tools: `Read`, `Edit`, `Bash`, `Search`, and `Fetch`. The first
three of which are implemented as sandboxed shell commands run through the sandbox.
`Read` is the simplest, with `cat -n … | sed -n …` providing line-numbered output with
the ability to extract a subset of lines in the file. `Bash` provides a standardized
tool calling interface with the ability to execute command-line tools within the
sandbox. `Edit` is the most complex, as agents require the ability to find-and-replace
*only when a unique match is found*. This is implemented in
`tart-agents/src/data/edit.pl`, which provides a custom stream editor similar to sed's
regex escape mode. Both the `Read` and `Edit` tools operate under the sandbox and are
thread-safe (meaning parallel agents cannot read or create partial modifications).

The web pair lives in `tart-agents/src/tools/web.rs` and runs outside the sandbox (see
above). `Search` shells out to the locally installed
[ddgs](https://github.com/deedy5/ddgs) CLI and returns a numbered list of title, url,
and snippet. `Fetch` reads one URL through the [r.jina.ai](https://r.jina.ai) reader
service, which returns the page as markdown, or directly with `raw=true` for JSON and
plain-text endpoints. Results are refused for non-public hosts. Each is offered to the
model only when its binary is installed. `TART_SEARCH_BIN` and `TART_FETCH_BIN` override
the binary lookups, and `TART_JINA_KEY` adds a bearer token if your reader account needs
one.

### Plan Mode

The keyboard combination `Shift+Tab` or `/plan` enter a read-only plan mode, allowing
the model to iterate on a design before applying any changes to the working directory.
To ensure the model has a scratch workspace for test commands, build scripts, and
similar, `/tmp` is still writable in plan mode. *These filesystem requirements are
enforced at the OS level by the sandbox!* The model is also provided a helpful temporary
[message](/tart-agents/src/data/PLAN.md) informing them of the plan mode restrictions
and how they should proceed. This message is stripped from context when plan mode is
exited to ensure future messages or compaction do not accidentally misinterpret the
planning state.

### User Conveniences

Files outside the sandbox can be natively added to the model's context with
`@file-mentions`, which supports a native file picker at arbitrary relative or absolute
paths(e.g. `@../other-repo/README.md` or `/Users/jenna/file.txt`). If the requested file
would be otherwise blocked by the sandbox, *tart* adds the file directly into context of
the containing user turn. Similarly `!bash` commands, which can only be initiated by the
user, place their output above the next turn's user message.

We use [`pulldown-cmark`](https://github.com/pulldown-cmark/pulldown-cmark/) to render
model responses with coloring and proper markdown styling, including nice table formats.
The style for this is kept in [`markdown.rs`](./tart-tui/src/pane/markdown.rs), and can
be extended or restyled as desired.

## Package Structure

Internally, we use an agent harness package structure similar to the one
[recommended by Huggingface](https://twotimespi.dev/internals/architecture/). This
separates the agent interface `tart-agents` (model client, transcript, bash tool,
sandbox) from the frontend code `tart-tui`. Currently, we support OpenAI Responses
endpoints, but we welcome PRs for a wider variety of providers.

## LLM Policy

PRs, issues, and code comments MUST be handwritten. You can use LLMs as part of your
development workflow, but I'd like a human to explain the goal of their changes.
