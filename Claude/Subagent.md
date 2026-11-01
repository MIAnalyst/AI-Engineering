# Subagent

Create and use specialized AI subagents in Claude Code for task-specific workflows and improved context management.

Subagents are specialized AI assistants that handle specific types of tasks. Each subagent runs in its own context window with a custom system prompt, specific tool access, and independent permissions. When Claude encounters a task that matches a subagent’s description, it delegates to that subagent, which works independently and returns results.

If you need multiple agents working in parallel and communicating with each other, see [agent teams](https://code.claude.com/docs/en/agent-teams) instead. Subagents work within a single session; agent teams coordinate across separate sessions.

Subagents help you:

- Preserve context by keeping exploration and implementation out of your main conversation
- Enforce constraints by limiting which tools a subagent can use
- Reuse configurations across projects with user-level subagents
- Specialize behavior with focused system prompts for specific domains
- Control costs by routing tasks to faster, cheaper models like Haiku

Claude uses each subagent’s description to decide when to delegate tasks. When you create a subagent, write a clear description so Claude knows when to use it.

Claude Code includes several built-in subagents like Explore, Plan, and general-purpose. You can also create custom subagents to handle specific tasks. This page covers the [built-in subagents](https://code.claude.com/docs/en/sub-agents#built-in-subagents), [how to create your own](https://code.claude.com/docs/en/sub-agents#quickstart-create-your-first-subagent), [full configuration options](https://code.claude.com/docs/en/sub-agents#configure-subagents), [patterns for working with subagents](https://code.claude.com/docs/en/sub-agents#work-with-subagents), and [example subagents](https://code.claude.com/docs/en/sub-agents#example-subagents).

## Built-in subagents

Claude Code includes built-in subagents that Claude automatically uses when appropriate. Each inherits the parent conversation’s permissions with additional tool restrictions.

### Explore

- A fast, read-only agent optimized for searching and analyzing codebases.
- Model: Haiku (fast, low-latency)
- Tools: Read-only tools (denied access to Write and Edit tools)
- Purpose: File discovery, code search, codebase exploration

Claude delegates to Explore when it needs to search or understand a codebase without making changes. This keeps exploration results out of your main conversation context.

When invoking Explore, Claude specifies a thoroughness level: quick for targeted lookups, medium for balanced exploration, or very thorough for comprehensive analysis.

You don't call Explore directly with a command. Instead, Claude uses each subagent's description to decide when to delegate tasks. So you simply describe what you need in natural language, and Claude Code will automatically spin up the Explore subagent when appropriate. For example:

- *"Find all API endpoints in this codebase"*
- *"Where is the authentication logic defined?"*
- *"Search for all files using the database connection"*

You can also nudge Claude to use Explore explicitly by saying something like "use the Explore agent to find…"

#### Thoroughness levels

When calling this agent, specify the desired thoroughness level: "quick" for basic searches, "medium" for moderate exploration, or "very thorough" for comprehensive analysis across multiple locations and naming conventions. This is something Claude Code decides internally when it delegates to Explore — it picks the thoroughness level based on how complex your request seems. You can influence it by how you phrase your prompt (e.g., "do a quick search for…" vs. "thoroughly explore all…").

#### Programmatic usage (subagent Task syntax)

If you're working with custom subagents or swarm orchestration, you can invoke Explore via the Task syntax, specifying the subagent type and optionally a model:

```
Task({
  subagent_type: "Explore",
  description: "Find API endpoints",
  prompt: "Find all API endpoints in this codebase. Be very thorough.",
  model: "haiku"  // optional: fast and cheap
})
```

### Plan

- A research agent used during [plan mode](https://code.claude.com/docs/en/common-workflows#use-plan-mode-for-safe-code-analysis) to gather context before presenting a plan.
- Model: Inherits from main conversation
- Tools: Read-only tools (denied access to Write and Edit tools)
- Purpose: Codebase research for planning

When you’re in plan mode and Claude needs to understand your codebase, it delegates research to the Plan subagent. This prevents infinite nesting (subagents cannot spawn other subagents) while still gathering necessary context.

### General_purpose

- A capable agent for complex, multi-step tasks that require both exploration and action.
- Model: Inherits from main conversation
- Tools: All tools
- Purpose: Complex research, multi-step operations, code modifications

Claude delegates to general-purpose when the task requires both exploration and modification, complex reasoning to interpret results, or multiple dependent steps.

Beyond these built-in subagents, you can create your own with custom prompts, tool restrictions, permission modes, hooks, and skills. The following sections show how to get started and customize subagents.

There are two built in agents:

- **claude-code-guide** (runs on Haiku) — answers questions about Claude Code itself, the Agent SDK, and the Anthropic API. It has read-only tools plus web fetch/search.
- **statusline-setup** (runs on Sonnet) — configures your Claude Code status line display settings.

The statusline-setup subagent configures the **status line** at the bottom of your Claude Code terminal — that thin bar that shows live info while you're working.

You can customize it to display things like your current git branch, Node version, active model, context usage, cost tracking, and other project metrics. Instead of manually editing config, you just describe what you want in natural language (e.g. "show me git branch and token cost in the status line") and the statusline-setup agent handles the configuration for you.

## Create your first subagent

Subagents are defined in Markdown files with YAML frontmatter. You can [create them manually](https://code.claude.com/docs/en/sub-agents#write-subagent-files) or use the /agents command. Project subagents (.claude/agents/) are ideal for subagents specific to a codebase.

The /agents command provides an interactive interface for managing subagents. Run /agents to:

- View all available subagents (built-in, user, project, and plugin)
- Create new subagents with guided setup or Claude generation
- Edit existing subagent configuration and tool access
- Delete custom subagents
- See which subagents are active when duplicates exist

This is the recommended way to create and manage subagents. For manual creation or automation, you can also add subagent files directly.

To list all configured subagents from the command line without starting an interactive session, run claude agents. This shows agents grouped by source and indicates which are overridden by higher-priority definitions.

Subagents are loaded at session start. If you create a subagent by manually adding a file, restart your session or use /agents to load it immediately.

Example of subagent

```
---
name: code-reviewer
description: Reviews code for quality and best practices
tools: Read, Glob, Grep
model: sonnet
---
You are a code reviewer. When invoked, analyze the code and provide
specific, actionable feedback on quality, security, and best practices.
```

The part between the --- delimiters is the YAML frontmatter — it defines the subagent's metadata and configuration. Everything below the closing --- becomes the system prompt. The body becomes the system prompt that guides the subagent’s behavior. Subagents receive only this system prompt (plus basic environment details like working directory), not the full Claude Code system prompt.

### Supported frontmatter fields

The following fields can be used in the YAML frontmatter. Only name and description are required.

| Field | Required | Description | Comment |
| --- | --- | --- | --- |
| name | Yes | Unique identifier using lowercase letters and hyphens |  |
| description | Yes | When Claude should delegate to this subagent |  |
| tools | No | [Tools](https://code.claude.com/docs/en/sub-agents#available-tools) the subagent can use. Inherits all tools if omitted |  |
| disallowedTools | No | Tools to deny, removed from inherited or specified list. | You can also prevent Claude fro using a built in ot custoer sub agents by mention them in the disallowedTools<br>claude --disallowedTools "Agent(Explore)" |
| model | No | [Model](https://code.claude.com/docs/en/sub-agents#choose-a-model) to use: sonnet, opus, haiku, a full model ID (for example, claude-opus-4-6), or inherit. Defaults to inherit | The model field controls which [AI model](https://code.claude.com/docs/en/model-config) the subagent uses:<br>- Model alias: Use one of the available aliases: sonnet, opus, or haiku<br>- Full model ID: Use a full model ID such as claude-opus-4-6 or claude-sonnet-4-6. Accepts the same values as the --model flag<br>- inherit: Use the same model as the main conversation<br>- Omitted: If not specified, defaults to inherit (uses the same model as the main conversation) |
| permissionMode | No | [Permission mode](https://code.claude.com/docs/en/sub-agents#permission-modes): default, acceptEdits, dontAsk, bypassPermissions, or plan | The default permission mode for a subagent is to **inherit from the parent session**. So whatever mode you're running in the main conversation (plan, ask before edit, edit automatically) carries over to the subagent. |
| maxTurns | No | Maximum number of agentic turns before the subagent stops |  |
| skills | No | Use the skills field to inject skill content into a subagent’s context at startup. This gives the subagent domain knowledge without requiring it to discover and load skills during execution.<br>The full content of each skill is injected into the subagent’s context, not just made available for invocation. Subagents don’t inherit skills from the parent conversation; you must list them explicitly.<br>So the sentence is saying: when you assign a skill to a subagent, the agent doesn't just get a reference it can call — it gets the **entire content loaded upfront** into its prompt. This matters because:<br>- It uses up context window space whether or not the agent needs it<br>- The agent always has that knowledge, so it can apply it passively (e.g. style guidelines) without explicitly "invoking" anything<br>- You need to be deliberate about which skills you assign, since each one eats into the available context | Subagents start with a **clean slate** for skills. Unlike tools and permissions which inherit from the parent by default, skills do not carry over at all.<br>So if your main session has access to certain skills, a subagent won't see any of them unless you explicitly list them in the frontmatter:<br>Example of giving subagent skills<br>---<br>name: frontend-builder<br>description: Builds frontend components<br>skills:<br>- /path/to/frontend-design/SKILL.md<br>- /path/to/another-skill/SKILL.md<br>--- |
| mcpServers | No | [MCP servers](https://code.claude.com/docs/en/mcp) available to this subagent. Each entry is either a server name referencing an already-configured server (e.g., "slack") or an inline definition with the server name as key and a full [MCP server config](https://code.claude.com/docs/en/mcp#configure-mcp-servers) as value |  |
| hooks | No | Subagents can define [hooks](https://code.claude.com/docs/en/hooks) that run during the subagent’s lifecycle. There are two ways to configure hooks:<br>- In the subagent’s frontmatter: Define hooks that run only while that subagent is active<br>- In settings.json: Define hooks that run in the main session when subagents start or stop<br>[Lifecycle hooks](https://code.claude.com/docs/en/sub-agents#define-hooks-for-subagents) scoped to this subagent<br>Hooks events such as:<br>\| Event \| Matcher input \| When it fires \|<br>\| --- \| --- \| --- \|<br>\| PreToolUse \| Tool name \| Before the subagent uses a tool \|<br>\| PostToolUse \| Tool name \| After the subagent uses a tool \|<br>\| Stop \| (none) \| When the subagent finishes (converted to SubagentStop at runtime) \| | This example creates a subagent that only allows read-only database queries. The PreToolUse hook runs the script specified in command before each Bash command executes:<br>```<br>---<br>name: db-reader<br>description: Execute read-only database queries<br>tools: Bash<br>hooks:<br> PreToolUse:<br> - matcher: "Bash"<br> hooks:<br> - type: command<br> command: "./scripts/validate-readonly-query.sh"<br>---<br>``` |
| memory | No | The memory field gives the subagent a persistent directory that survives across conversations. The subagent uses this directory to build up knowledge over time, such as codebase patterns, debugging insights, and architectural decisions.<br>project is the recommended default scope. It makes subagent knowledge shareable via version control. Use user when the subagent’s knowledge is broadly applicable across projects, or local when the knowledge should not be checked into version control. | The default is **no memory** — if you don't specify it, the subagent won't read or write to any MEMORY.md files. Each invocation starts fresh with no cross-session learning.<br>Ask the subagent to consult its memory before starting work: “Review this PR, and check your memory for patterns you’ve seen before.”<br>Ask the subagent to update its memory after completing a task: “Now that you’re done, save what you learned to your memory.” Over time, this builds a knowledge base that makes the subagent more effective.<br>Include memory instructions directly in the subagent’s markdown file so it proactively maintains its own knowledge base:<br>```<br>Update your agent memory as you discover codepaths, patterns, library<br>locations, and key architectural decisions. This builds up institutional<br>knowledge across conversations. Write concise notes about what you found<br>and where.<br>``` |
| background | No | Set to true to always run this subagent as a [background task](https://code.claude.com/docs/en/sub-agents#run-subagents-in-foreground-or-background). Default: false |  |
| isolation | No | Set to worktree to run the subagent in a temporary [git worktree](https://code.claude.com/docs/en/common-workflows#run-parallel-claude-code-sessions-with-git-worktrees), giving it an isolated copy of the repository. The worktree is automatically cleaned up if the subagent makes no changes |  |

## Control subagent capabilities

You can control what subagents can do through tool access, permission modes, and conditional rules.

### Available tools

Subagents can use any of Claude Code’s [internal tools](https://code.claude.com/docs/en/tools-reference). By default, subagents inherit all tools from the main conversation, including MCP tools. To restrict tools, use the tools field (allowlist) or disallowedTools field (denylist):

---￼name: safe-researcher￼description: Research agent with restricted capabilities￼tools: Read, Grep, Glob, Bash￼disallowedTools: Write, Edit￼---￼

### Restrict which subagents can be spawned

When an agent runs as the main thread with claude --agent, it can spawn subagents using the Agent tool. To restrict which subagent types it can spawn, use Agent(agent_type) syntax in the tools field.  
---￼name: coordinator￼description: Coordinates work across specialized agents￼tools: Agent(worker, researcher), Read, Bash￼---￼  
This is an allowlist: only the worker and researcher subagents can be spawned. If the agent tries to spawn any other type, the request fails and the agent sees only the allowed types in its prompt. To block specific agents while allowing all others, use [permissions.deny](https://code.claude.com/docs/en/sub-agents#disable-specific-subagents) instead. To allow spawning any subagent without restrictions, use Agent without parentheses:  
tools: Agent, Read, Bash￼

If Agent is omitted from the tools list entirely, the agent cannot spawn any subagents. This restriction only applies to agents running as the main thread with claude --agent. Subagents cannot spawn other subagents, so Agent(agent_type) has no effect in subagent definitions. There's a hard rule: no nesting. A subagent is always a leaf node — it does its work and returns results. It can never delegate further. So even if you add Agent to a subagent's tools list, it's ignored.

Meaning of "agents running as the main thread with claude --agent"  
Normally when you start Claude Code, you just run: claude. This starts the default Claude Code experience — the built-in orchestrator that can spawn Explore, Plan, general-purpose subagents, etc. But you can replace that default orchestrator with your own custom agent:  
claude --agent my-custom-agent

I reached: Scope MCP servers to a subagent

Source: [https://code.claude.com/docs/en/sub-agents](https://code.claude.com/docs/en/sub-agents)  
Source: [https://code.claude.com/docs/en/sub-agents#enable-persistent-memory](https://code.claude.com/docs/en/sub-agents#enable-persistent-memory)
