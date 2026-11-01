# Interesting Prompt

- Isolate high-volume operations: Use a subagent to run the test suite and report only the failing tests with their error messages
- Run parallel research: Research the authentication, database, and API modules in parallel using separate subagents
- Invoke subagents explicitly: "description": "Expert code reviewer. Use proactively after code changes."
- Use the code-reviewer subagent to find performance issues, then use the optimizer subagent to fix them
- /btw what did we rename the auth function to?
