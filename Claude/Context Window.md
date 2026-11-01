# Context Window

Recommendation of how to deal with:

**Keep conversations focused.** Rather than one mega-long chat covering many topics, start a new conversation when you switch to a different task. This keeps the context clean and relevant.

**Front-load important information.** Put the most critical details — instructions, constraints, key data — early in your message. If the conversation gets long, material at the beginning and end tends to be referenced more reliably than stuff buried in the middle.

**Summarize as you go.** If a conversation is getting lengthy, you can ask me to summarize what we've covered so far. Then you can start a fresh chat, paste that summary in, and continue without losing the thread.

**Be explicit about what matters.** Don't assume I'll "just know" something from 50 messages ago. If a detail is important for the current question, restate it. A quick reminder like "as we discussed, the budget is $10k" goes a long way.

**Chunk large documents.** If you're working with a big file or dataset, focus on one section at a time rather than dumping everything in at once.

**Use system-level features.** If you're building with the API, you can use the system prompt to hold persistent instructions so they don't eat into your conversation space. On claude.ai, you can set preferences in your settings that persist across chats.

**Structure your inputs.** Clear formatting — labels, headers, numbered lists — helps me parse your input more efficiently, which means better use of the available window.

The core principle is: treat the context window like a whiteboard with limited space. Keep it organized, erase what you don't need, and make sure the important stuff is always visible.

Running /clear in Claude Code wipes the conversation history within the same session, giving you a fresh context window without needing to open a new chat tab.

**/compact** — Smart summary. Condenses the conversation but preserves important details like what files you're working on and decisions you've made. Use this when you want to free up space but keep working on the same task.
