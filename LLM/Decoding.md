# Decoding

Here is lightly touching about decoding but I need to read more about it.

**Decoding** is the final step of generating text: turning the model's probability distribution into one actual token. The transformer's job ends by producing a probability for every token in its vocabulary — "Warsaw" 61%, "Krakow" 12%, "the" 3%, and so on across all 50,000. But you can't output a distribution. Something has to choose. That choosing is decoding.

It runs outside the neural network, as ordinary code. Same model, different decoding settings, noticeably different output — which is why temperature is a slider you can change rather than something requiring retraining.

The main strategies:

- **Greedy** — always take the highest. Deterministic, tends toward flat and repetitive.
- **Beam search** — keep several candidate sequences alive and score them as wholes. Standard in translation; largely dropped for chat because it produces bland text.
- **Sampling** — used in LLM chat - The model produces probabilities over the whole vocabulary, and then a token is picked *randomly*, weighted by those probabilities. Not the top one. Not a beam. This is why you get a different answer when you ask the same question twice. That's sampling, not the model changing its mind. Sampling is tuned with:
    - **Temperature** — flattens or sharpens the probability distribution. Low means more predictable, high means more varied. Around 0 it collapses back to greedy.
    - **Top-p (nucleus sampling)** — keep only the most likely tokens that together cover, say, 90% of the probability mass, then sample from those. Cuts off the long tail of nonsense.
    - **Top-k** — same idea, but keep a fixed number of candidates.

Top-p plus temperature is the common default.
