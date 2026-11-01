# Claude Memory

## How Claude remembers your project

Give Claude persistent instructions with CLAUDE.md files, and let Claude accumulate learnings automatically with auto memory.

Each Claude Code session begins with a fresh context window. Two mechanisms carry knowledge across sessions:

- CLAUDE.md files: instructions you write to give Claude persistent context
- Auto memory: notes Claude writes itself based on your corrections and preferences

### CLAUDE.md files

CLAUDE.md files are markdown files that give Claude persistent instructions for a project, your personal workflow, or your entire organization. You write these files in plain text; Claude reads them at the start of every session.

A project CLAUDE.md can be stored in either ./CLAUDE.md or ./.claude/CLAUDE.md. Create this file and add instructions that apply to anyone working on the project: build and test commands, coding standards, architectural decisions, naming conventions, and common workflows. These instructions are shared with your team through version control, so focus on project-level standards rather than personal preferences. ---> The .claude/ location is actually the recommended place to keep it alongside other Claude-specific config.

Run /init to generate a starting CLAUDE.md automatically. Claude analyzes your codebase and creates a file with build commands, test instructions, and project conventions it discovers. If a CLAUDE.md already exists, /init suggests improvements rather than overwriting it. Refine from there with instructions Claude wouldn’t discover on its own.

Size: target under 200 lines per CLAUDE.md file. Longer files consume more context and reduce adherence. If your instructions are growing large, split them using [imports](https://code.claude.com/docs/en/memory#import-additional-files) or [.claude/rules/](https://code.claude.com/docs/en/memory#organize-rules-with-clauderules) files.

#### Import additional files

CLAUDE.md files can import additional files using @path/to/import syntax. Imported files are expanded and loaded into context at launch alongside the CLAUDE.md that references them. Example: @../Docs/rag_end_to_end.md

For personal preferences you don’t want to check in, import a file from your home directory. The import goes in the shared CLAUDE.md, but the file it points to stays on your machine:  
# Individual Preferences  
- @~/.claude/my-project-instructions.md

#### Organize rules with .claude/rules/

For larger projects, you can organize instructions into multiple files using the .claude/rules/ directory. This keeps instructions modular and easier for teams to maintain. Rules can also be [scoped to specific file paths](https://code.claude.com/docs/en/memory#path-specific-rules), so they only load into context when Claude works with matching files, reducing noise and saving context space.

Rules load into context every session or when matching files are opened. For task-specific instructions that don’t need to be in context all the time, use [skills](https://code.claude.com/docs/en/skills) instead, which only load when you invoke them or when Claude determines they’re relevant to your prompt.

**Set up rules**  
Place markdown files in your project’s .claude/rules/ directory. Each file should cover one topic, with a descriptive filename like testing.md or api-design.md. All .md files are discovered recursively, so you can organize rules into subdirectories like frontend/ or backend/:

your-project/  
├── .claude/  
│ ├── CLAUDE.md # Main project instructions  
│ └── rules/  
│ ├── code-style.md # Code style guidelines  
│ ├── testing.md # Testing conventions  
│ └── security.md # Security requirements

Rules without [paths frontmatter](https://code.claude.com/docs/en/memory#path-specific-rules) are loaded at launch with the same priority as .claude/CLAUDE.md.

**Path-specific rules**  
Rules can be scoped to specific files using YAML frontmatter with the paths field. These conditional rules only apply when Claude is working with files matching the specified patterns.

.claude/rules/rag-rules.md

```
# .claude/rules/rag-rules.md
---
paths:
  - "rag/**/*.py"
  - "services/retriever/**/*.js"
---
# RAG-Specific Rules
- Chunk size: 500 tokens with 50 token overlap
- Use hybrid search (keyword + vector)
- Cache frequent queries for 5 minutes
```

Rules without a paths field are loaded unconditionally and apply to all files. Path-scoped rules trigger when Claude reads files matching the pattern, not on every tool use.

Use glob patterns in the paths field to match files by extension, directory, or any combination:

| Pattern | Matches |
| --- | --- |
| \*\*/\*.ts | All TypeScript files in any directory |
| src/\*\*/\* | All files under src/ directory |
| \*.md | Markdown files in the project root |
| src/components/\*.tsx | React components in a specific directory |

Difference between skills and Path-Specific Rules:

| Aspect | Path-Specific Rules | Skills |
| --- | --- | --- |
| Loading | Automatically loaded when matching files are opened | Only metadata (~100 tokens) kept in context; full content loaded on invocation |
| Activation | Reactive to file context — triggered by where you are in the codebase | Reactive to task context — triggered by what you're trying to do |
| Best suited for | Constraints, conventions, and style enforcement ("always/never do X here") | Procedures, workflows, and domain knowledge ("here's how to do X") |
| Context cost | Scales with number of open files; every matching file adds rule overhead | Cheap at rest; pays cost only when invoked, so can carry richer payloads (examples, multi-step guides, reference files) |
| Trade-off | Must stay concise to be practical, since they're always loaded | Can be detailed and comprehensive, since loading is amortized |
| Example | "All files in /api must use the standard error-handling wrapper" | "Here's how to write and run a database migration" |

### Auto memory

Auto memory lets Claude accumulate knowledge across sessions without you writing anything. Claude saves notes for itself as it works: build commands, debugging insights, architecture notes, code style preferences, and workflow habits. Claude doesn’t save something every session. It decides what’s worth remembering based on whether the information would be useful in a future conversation.

Auto memory is on by default. To toggle it, open /memory in a session and use the auto memory toggle,

Storage location  
Each project gets its own memory directory at ~/.claude/projects/\<project>/memory/. The \<project> path is derived from the git repository, so all worktrees and subdirectories within the same repo share one auto memory directory. Outside a git repo, the project root is used instead.

The directory contains a MEMORY.md entrypoint and optional topic files:

~/.claude/projects/\<project>/memory/  
├── MEMORY.md # Concise index, loaded into every session  
├── debugging.md # Detailed notes on debugging patterns  
├── api-conventions.md # API design decisions  
└── ... # Any other topic files Claude creates

MEMORY.md acts as an index of the memory directory. Claude reads and writes files in this directory throughout your session, using MEMORY.md to keep track of what’s stored where.

The first 200 lines of MEMORY.md are loaded at the start of every conversation. Content beyond line 200 is not loaded at session start. Claude keeps MEMORY.md concise by moving detailed notes into separate topic files.

When you ask Claude to remember something, like “always use pnpm, not npm” or “remember that the API tests require a local Redis instance,” Claude saves it to auto memory. press # during a session to quickly add a new memory entry without opening the full editor. Example: # remember to add comments to the code

Source: [https://code.claude.com/docs/en/memory#path-specific-rules](https://code.claude.com/docs/en/memory#path-specific-rules)
