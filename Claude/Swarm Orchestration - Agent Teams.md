# Swarm Orchestration / Agent Teams

Swarm orchestration (officially called **agent teams** by Anthropic) is a feature in Claude Code that lets you coordinate multiple Claude Code instances working together on a task in parallel, rather than relying on a single agent doing everything sequentially.

## The core idea

LLMs perform worse as context expands. The more information in the context window, the harder it is for the model to focus on what matters right now. Swarm orchestration addresses this by splitting complex work across multiple specialized agents, each with its own fresh, focused context window.

## How it works

You talk to a team lead — the lead agent doesn't write code itself. It plans, delegates, and synthesizes. When you approve a plan, the lead enters "delegation mode" and creates agents for specific roles: frontend, backend, testing, documentation, architecture. Each specialist gets a fresh context window focused on their task. They share a task board with dependency tracking and coordinate via an inbox-based messaging system.

For example, if you say "Users report checkout fails intermittently," the leader might spawn a log-analyst, a code-archaeologist, a reproducer, and a db-detective — all working independently and reporting findings back to the leader.

## When it's useful

The strongest use cases are research and review (multiple teammates investigating different aspects simultaneously), new modules or features (teammates each owning a separate piece), debugging with competing hypotheses (testing different theories in parallel), and cross-layer coordination (changes that span frontend, backend, and tests).

## When it's not

Agent teams add coordination overhead and use significantly more tokens than a single session. They work best when teammates can operate independently. For sequential tasks, same-file edits, or work with many dependencies, a single session or subagents are more effective.

## How to enable it

Agent teams are experimental and disabled by default. Enable them by adding CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS to your settings.json or environment. So in short: swarm orchestration is the jump from "one AI assistant doing everything" to "a team of AI specialists collaborating in parallel" — the same pattern as a real engineering team with a lead architect, implementers, and reviewers.

|  | Subagents | Agent teams (swarm) |
| --- | --- | --- |
| Architecture | *Lightweight, hierarchical* | *Distributed, parallel* |
| Model | Parent delegates a focused task to a child agent *within the same session*. Child reports back, parent continues. | A team lead spawns multiple *independent Claude Code instances*, each with its own context window and process. |
| Analogy | Asking a colleague to look something up and report back to you. | Assembling a project team where each person works on their part simultaneously. |
| Context | Shares the parent's context. Results merge back into the main conversation. | Each teammate has a fresh, isolated context window. No shared memory. |
| Execution | Sequential — one subagent at a time. | Parallel — multiple teammates work simultaneously. |
| Communication | One-way: child → parent only. Subagents can't talk to each other. | Multi-directional: teammates message each other and the lead via inboxes. |
| Configuration |  |  |
| Built-in types | Explore *(read-only search)*, Plan *(architecture)*, general-purpose *(read + write)* | None built-in — you define roles per project *(frontend, tester, security, etc.)* |
| Custom agents | Yes — define your own in .claude/subagents/ with a name, description, allowed tools, and system prompt. | Yes — define in team config with role, model, prompt, and tool access per teammate. |
| Setup effort | Minimal — built-in ones work automatically; custom ones need a config file. | More involved — enable experimental flag, define team structure, manage lifecycle and cleanup. |
| Trade-offs |  |  |
| Token cost | Lower — runs within existing session. | Significantly higher — each teammate is a full session. |
| Speed | Fast for focused tasks; bottlenecked on large jobs by sequential execution. | Slower startup (each agent loads context), but faster throughput on parallelizable work. |
| Coordination | Simple — parent decides what to delegate next. | Rich — shared task board with dependency tracking, self-claiming, and blocking. |
| File conflicts | Not a concern — only one agent edits at a time. | Handled via git worktree isolation and file locking. |
| When to use |  |  |
| Best for | Quick file searches • Gathering context before planning • Keeping verbose output out of main context • Focused, single-purpose tasks | Multi-file refactors • Parallel research or debugging • Cross-layer changes (frontend + backend + tests) • Competing-hypothesis investigations |
| Avoid when | Work needs true parallelism or cross-agent communication. | Task is sequential, touches one file, or steps depend heavily on each other. |

Sources:  
[https://code.claude.com/docs/en/agent-teams](https://code.claude.com/docs/en/agent-teams)
