# Neurons, weights, parameters

**Neurons never see text.** By the time a vector reaches a neuron, the text is long gone. The neuron sees 12,288 numbers. It has no idea they once meant "archived."

**Neurons don't calculate weights.** The weights are already there, frozen since training ended. The neuron *uses* its weights to calculate a score.

**What one neuron actually does**  
A neuron has a fixed list of 12,288 numbers (for example) — its weights. Think of it as the pattern it's looking for.

1. Multiply each incoming number by its matching weight.
2. Add all 12,288 results into a single number — the score.
3. Pass that score through a simple function that zeros out weak scores.
4. If it survived, the neuron's *output* weights get added to the word vector, scaled by the score.

So: **match on the way in, write on the way out.** Not "compare to text," but "compare to the pattern in my weights."

**The neuron's output weights are parameters.** That fixed list of 12,288 numbers sitting in the neuron, ready to be written out, was learned in training and never changes. Parameters.

**The general rule**  
Two categories of number in a running model:

- [ ] **Parameters** — frozen. Weights and biases. Same for every input, every user, forever.

**Activations** — temporary. Every number computed during a forward pass: the word vectors, the queries and keys, the match scores, the values added to the vector. Created for your input, discarded after.
