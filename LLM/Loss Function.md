# Loss Function

A loss function is a mathematical function that measures how wrong a model’s prediction is compared to the true target. During training, the model adjusts its parameters to minimize this value—so the loss is the signal that tells the model how to improve.  
Simple analogy:  
If the model predicts 42 but the correct answer is 50, the loss function outputs a number representing that error (e.g., 64 if using squared error). The optimizer then nudges the model’s weights to reduce that error next time.
