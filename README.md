# *tart*

*tart*: like a pi you can see inside! *tart* is a lightweight, auditable agent harness
implemented in a few thousand lines of code. Though minimal, *tart* implements a variety
of nice-to-have features including native tmux scrollback, manual tool calls via
`@filename`, inline bash commands via `!script.sh`, and subagents. Tool calls are
sandboxed by default, so tools can run without approval.

## Architecture

### Sandboxing

Sandboxing is *always* enabled for tart, and there is no way for the model to request
outside access or edit files outside the sandbox. As a result, all bash commands and
filesystem tools run without approval. In practice, this means that the model is given
unrestricted access to both the working directory, `/tmp`, and `$TMPDIR`, so plan
accordingly.

We choose `sandbox-exec` as a lightweight sandbox option, as it effectively balances
safety with memory usage and complexity. While running the agent in a container seems
beneficial, the real safety bottleneck is syncing local data with the agent's copies.
Manual approvals make this step safe in theory, but agentic workloads are capable of
generating far more requests for approval of complex commands then can reasonably be
evaluated. *Sandboxing is helpful, but you should always use version control and backup
tools if your working directory contains data that is not stored elsewhere!*

### Tools

*tart* provides three basic tools: `Read`, `Edit`, and `Bash`, each of which is
implemented as a sandboxed shell command run through the sandbox. `Read` is the
simplest, with `cat -n … | sed -n …` providing line-numbered output with the ability to
extract a subset of lines in the file. `Bash` provides a standardized tool calling
interface with the ability to execute command-line tools within the sandbox. `Edit` is
the most complex, as agents require the ability to find-and-replace *only when a unique
match is found*. This is implemented in `tart-agents/src/data/edit.pl`, which provides a
custom stream editor similar to sed's regex escape mode. Both the `Read` and `Edit`
tools operate under the sandbox and are thread-safe (meaning parallel agents cannot read
or create partial modifications).

## Package Structure

Internally, we use an agent harness package structure similar to the one
[recommended by Huggingface](https://twotimespi.dev/internals/architecture/). This
separates the agent interface `tart-agents` (model client, transcript, bash tool,
sandbox) from the frontend code `tart-tui`. Currently, we support OpenAI Responses
endpoints, but we welcome PRs for a wider variety of providers.

## LLM Policy

PRs, issues, and code comments MUST be handwritten. You can use LLMs as part of your
development workflow, but I'd like a human to explain the goal of their changes.
