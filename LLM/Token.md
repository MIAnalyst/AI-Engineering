# Token

A word is a linguistic unit. A token is a computational unit used by a language model. The model does not directly process words; it processes tokens.

**Core Difference**

| Aspect | Word | Token |
| --- | --- | --- |
| Definition | A unit of meaning/grammar, often separated by spaces in English. | A chunk of text produced by a tokenizer. |
| Granularity | Usually larger. | Can be a whole word, part of a word, a character, punctuation, or whitespace. |
| Example | "unbelievable" is one word. | "un", "believ", "able" — three tokens. |
| Count | Based on language rules. | Based on the model's fixed vocabulary and tokenizer algorithm. |
| Vector representation | Word2Vec gives one static vector per word. | An LLM gives one contextual vector per token. |

**Examples**

- "unbelievable" → 1 word, but often 3 tokens: un, believ, able
- "don't" → 1 word, but often 2 tokens: do, n't
- "New York" → 2 words, and usually 2 tokens: New, York
- " " (a space) → not a word, but it can be part of a token or its own token.
- "ChatGPT" → 1 word, but may be split into Chat, G, PT depending on the tokenizer.

**Why Tokenization Exists**

Models need a fixed vocabulary. If every word were a token, the vocabulary would be enormous and rare words would be impossible to handle.

Instead, tokenizers use algorithms like BPE (Byte Pair Encoding), WordPiece, or SentencePiece to break text into frequent subword units.

Common words (words that are often written) often become one token. Rare words (words rarely written) get split into smaller pieces. This helps the model handle new words, typos, morphology, and multiple languages.

The key idea is that a tokenizer is not a dictionary of words. It is a compression system that learns which character sequences are frequent enough to deserve their own token. The model then works with those tokens, not words.
