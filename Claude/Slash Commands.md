# Slash Commands

| Command | What It Does | When to Use It |
| --- | --- | --- |
| /help | Lists all available commands (built-in + custom) | When you can't remember a command or want to see what's available |
| /init | Creates a CLAUDE.md file for your project with context about your codebase | Once at the start of any new project — it helps Claude understand your project |
| /clear | Wipes conversation history completely, resets context window to full capacity | When switching to a completely different task or quality is degrading |
| /compact | Summarizes conversation to free up context space while keeping key info. Accepts optional instructions like /compact "focus on the API work" | Mid-task when the context is filling up but you still need continuity |
| /context | Shows a visual breakdown of what's consuming your context window | To check how full your context is and decide if you need /compact or /clear |
| /cost | Displays token usage and estimated session cost in dollars | To monitor spending and consumption during a session |
| /model | Switch between Claude models (Sonnet, Opus, Haiku) | When you want a faster/cheaper model for simple tasks, or a more powerful one for complex work |
| /review | Requests a code review of recent changes | After making changes, before committing |
| /memory | Opens your CLAUDE.md file for editing | When you need to update project-level instructions or preferences |
| /permissions | Manage tool permissions (read/write/execute) | To allow or restrict what Claude can do in your project |
| /config | Opens the interactive settings interface | To adjust permissions, theme, default model, integrations |
| /doctor | Diagnoses installation and connection issues | When something isn't working right |
| /login | Authenticate or re-authenticate | When your session expires or you need to switch accounts |
| /logout | Disconnect your session | When ending your work session |
| /status | Shows current session status | To check connection, model, and session health |
| /add-dir | Adds additional directories to Claude's context | When working across multiple repositories |
| /hooks | Opens the interactive hooks management interface | To set up automated actions that trigger on events |
| /mcp | Manage MCP (Model Context Protocol) server connections | To connect external tools like GitHub, databases, APIs |
| /plugins | Manage installed plugins and marketplaces | To extend Claude Code with community tools |
| /resume | Resume a previous conversation by ID or name | To pick up where you left off in a past session |
| /terminal-setup | Installs Shift+Enter keybinding for your terminal | One-time setup for better line-break behavior |

A useful combo to remember: check /cost, then /compact to compress, then /cost again to see how much you saved. [SFEIR Institute](https://institute.sfeir.com/en/claude-code/claude-code-essential-slash-commands/faq/)  
And remember — on top of these built-in commands, you can create your own custom commands by placing markdown files in .claude/commands/ (project-level) or ~/.claude/commands/ (personal, across all projects)
