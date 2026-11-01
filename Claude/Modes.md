# Modes

| VS Code Label | CLI Equivalent | What It Does |
| --- | --- | --- |
| **Ask before edit** | Default Mode | Claude can read and edit files, but asks your permission before every change |
| **Edit automatically** | Auto-Accept Mode | Claude makes file edits without asking — faster, but less control |
| **Plan mode** | Plan Mode | Read-only — Claude explores and plans but can't modify anything |

## The Three Modes

### 1. Default Mode (Normal/Edit Mode)

This is what you start with. Every change requires confirmation. The safest choice for beginners and careful work. Claude can read files, write files, and run commands — but it asks your permission before doing anything that modifies your project.  
**When to use it:** Day-to-day work where you want to stay in control. Especially good when you're still learning Claude Code or working on sensitive parts of your codebase.

### 2. Auto-Accept Mode

Claude edits files without asking, but still asks for commands. It speeds things up significantly because you skip all the "approve this edit?" prompts.  
**When to use it:** When you trust the plan and want Claude to execute quickly — for example, after you've already reviewed a plan, or for repetitive tasks like adding tests or formatting code. The creator of Claude Code himself uses this workflow: start in Plan Mode, go back and forth until the plan is right, then switch to Auto-Accept and let Claude execute.

### 3. Plan Mode

In Plan Mode, Claude has access to read-only and research tools only. It can explore your codebase, search files, and analyze code, but it cannot edit anything. It produces a structured plan that you review before any code gets written.  
**When to use it:** Before tackling complex, multi-file changes. Claude analyzes your entire codebase, identifies dependencies, suggests strategies, and estimates complexity — all before touching code. Also great for understanding unfamiliar codebases.

## The Recommended Workflow

The most effective pattern is: **Plan → Review → Execute**

1. Start in **Plan Mode** — ask Claude to analyze the problem
2. Review and refine the plan together
3. Switch to **Auto-Accept Mode** — let Claude implement the approved plan
4. Switch back to **Default Mode** for any fine-tuning

## Do Slash Commands Need Specific Modes?

No — slash commands work independently of modes. You can run /compact, /cost, /help, /model, etc. in any mode. The mode only controls what Claude is *allowed to do* with your files:

- In **Plan Mode** , Claude can still respond to /compact, /cost, /context, etc. — it just can't edit your code.
- In **Auto-Accept** , all commands work the same, but Claude won't pause for edit confirmations.
- In **Default** , everything works with the normal approval prompts.

The only real exception is /plan, which is specifically for entering Plan Mode. Otherwise, think of modes and slash commands as two separate things — modes control permissions, slash commands trigger actions
