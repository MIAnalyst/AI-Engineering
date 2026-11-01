# Hooks

Hooks are user-defined shell commands that execute at specific points in Claude Code’s lifecycle. They provide deterministic control over Claude Code’s behavior, ensuring certain actions always happen rather than relying on the LLM to choose to run them. Use hooks to enforce project rules, automate repetitive tasks, and integrate Claude Code with your existing tools.

Hooks are functions that let you "hook into" and execute custom code at specific points in a process or lifecycle. They allow you to extend or modify behavior without changing the core code. Think of them like "trigger points" where you can insert your own logic.

Why Use Hooks?

1. Separation of concerns - Keep related code together
2. Extensibility - Add features without modifying core code
3. Reusability - Use same hook logic in multiple places
4. Lifecycle management - Execute code at specific time

For decisions that require judgment rather than deterministic rules, you can also use [prompt-based hooks](https://code.claude.com/docs/en/hooks-guide#prompt-based-hooks) or [agent-based hooks](https://code.claude.com/docs/en/hooks-guide#agent-based-hooks) that use a Claude model to evaluate conditions.

### Prompt-based hooks

For decisions that require judgment rather than deterministic rules, use type: "prompt" hooks. Instead of running a shell command, Claude Code sends your prompt and the hook’s input data to a Claude model (Haiku by default) to make the decision. You can specify a different model with the model field if you need more capability.

Example

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "prompt",
            "prompt": "Check if all tasks are complete. If not, respond with {\"ok\": false, \"reason\": \"what remains to be done\"}."
          }
        ]
      }
    ]
  }
}
```

### Agent-based hooks

When verification requires inspecting files or running commands, use type: "agent" hooks. Unlike prompt hooks which make a single LLM call, agent hooks spawn a subagent that can read files, search code, and use other tools to verify conditions before returning a decision.

Example

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "agent",
            "prompt": "Verify that all unit tests pass. Run the test suite and check the results. $ARGUMENTS",
            "timeout": 120
          }
        ]
      }
    ]
  }
}
```
