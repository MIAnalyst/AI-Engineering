# Work with subagents

## Automatic delegation

Claude automatically delegates tasks based on the task description in your request, the description field in subagent configurations, and current context. To encourage proactive delegation, include phrases like “use proactively” in your subagent’s description field.

Examples:

- "description": "Expert code reviewer. Use proactively after code changes."
- "description": "Debugging specialist for errors, test failures, and unexpected behavior. Use proactively when encountering any issues."

## Invoke subagents explicitly

When automatic delegation isn’t enough, you can request a subagent yourself. Three patterns escalate from a one-off suggestion to a session-wide default:

- Natural language: name the subagent in your prompt; Claude decides whether to delegate
- @-mention: guarantees the subagent runs for one task
- Session-wide: the whole session uses that subagent’s system prompt, tool restrictions, and model via the --agent flag or the agent setting

Example for natural language: Use the test-runner subagent to fix failing tests  
@-mention the subagent: @"code-reviewer (agent)" look at the auth changes

Run the whole session as a subagent: claude --agent \<name>  
The subagent’s system prompt replaces the default Claude Code system prompt entirely, the same way [--system-prompt](https://code.claude.com/docs/en/cli-reference) does. --system-prompt replaces the entire default system prompt with custom text. It completely overrides Claude Code's built-in instructions. For example: claude --system-prompt "You are a Python expert"  
To make it the default for every session in a project, set agent in .claude/settings.json

## Run subagents in foreground or background

Subagents can run in the foreground (blocking) or background (concurrent):

- Foreground subagents block the main conversation until complete. Permission prompts and clarifying questions (like [AskUserQuestion](https://code.claude.com/docs/en/tools-reference) ) are passed through to you.
- Background subagents run concurrently while you continue working. Before launching, Claude Code prompts for any tool permissions the subagent will need, ensuring it has the necessary approvals upfront. Once running, the subagent inherits these permissions and auto-denies anything not pre-approved. If a background subagent needs to ask clarifying questions, that tool call fails but the subagent continues.

If a background subagent fails due to missing permissions, you can start a new foreground subagent with the same task to retry with interactive prompts.

Claude decides whether to run subagents in the foreground or background based on the task. You can also:

- Ask Claude to “run this in the background”
- Press Ctrl+B to background a running task. The main agent (your primary conversation) always runs in the foreground and cannot be backgrounded.

## Effective subagent use cases

### Isolate high-volume operations

One of the most effective uses for subagents is isolating operations that produce large amounts of output. Running tests, fetching documentation, or processing log files can consume significant context. By delegating these to a subagent, the verbose output stays in the subagent’s context while only the relevant summary returns to your main conversation. Example:  
*Use a subagent to run the test suite and report only the failing tests with their error messages*

### Run parallel research

For independent investigations, spawn multiple subagents to work simultaneously. Example:  
*Research the authentication, database, and API modules in parallel using separate subagents**￼*

Each subagent explores its area independently, then Claude synthesizes the findings. This works best when the research paths don’t depend on each other.

When subagents complete, their results return to your main conversation. Running many subagents that each return detailed results can consume significant context.

For tasks that need sustained parallelism or exceed your context window, [agent teams](https://code.claude.com/docs/en/agent-teams) give each worker its own independent context.

## Chain subagents

For multi-step workflows, ask Claude to use subagents in sequence. Each subagent completes its task and returns results to Claude, which then passes relevant context to the next subagent. Example:  
*Use the code-reviewer subagent to find performance issues, then use the optimizer subagent to fix them*

## Choose between subagents and main conversation

Use the main conversation when:

- The task needs frequent back-and-forth or iterative refinement
- Multiple phases share significant context (planning → implementation → testing)
- You’re making a quick, targeted change
- Latency matters. Subagents start fresh and may need time to gather context

Use subagents when:

- The task produces verbose output you don’t need in your main context
- You want to enforce specific tool restrictions or permissions
- The work is self-contained and can return a summary

Consider [Skills](https://code.claude.com/docs/en/skills) instead when you want reusable prompts or workflows that run in the main conversation context rather than isolated subagent context.

For a quick question about something already in your conversation, use [/btw](https://code.claude.com/docs/en/interactive-mode#side-questions-with-btw) instead of a subagent. It sees your full context but has no tool access, and the answer is discarded rather than added to history.  
**Example:** You're in the middle of a long refactoring session and you think "wait, what was the name of that function we renamed earlier?"

- You type /btw what did we rename the auth function to?
- Claude can answer because it **sees your full conversation history**
- But the question and answer **disappear** — they don't get added to the conversation
- And it **can't run code or edit files** — it's purely a Q&A
