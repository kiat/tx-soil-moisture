# Attention Mechanism 
## Alexander Hoang

"""
This work begins with an implementation of the feature importance analysis. It also contains some jupyter notebooks on Attention Mechanism inspired by another implementation here - https://dlsyscourse.org/lectures/.

Implemented a simple self-attention model as well as a transformer model and saw how it performed in 
365 days of data of station1 dataset. It performs pretty well.


TRANSFORMER - inspired from https://github.com/dlsyscourse/public_notebooks/blob/main/rnn_implementation.ipynb
The model has an initial linear layer (input_proj) that projects to the  hidden dimension (hidden_dim). 

A TransformerEncoderLayer with multi-head attention (n_heads) and feed-forward dimensions (dim_feedforward).
The layer uses causal masking to prevent future information from leaking into past predictions as shown in the tutorial above.

Output Layer:
The output_layer maps the transformed output to the target dimension (however many hours), predicting a single value per sample.


Training:
* MSE is used as the loss function
* MAE, MAPE, MSE are all calculated for each batch and averaged at the end

Validation:
* Evaluation occurs at the end of the epoch to see the validation loss

Testing:
* Average MSE, MAE, MAPE are all averaged

Graphs:
* Prediction vs. Actual. Sees how well the model performs. It compares the model predictions with actual target values
* training loss vs validation loss: sees if we're overfitting if the lines diverge. They don't



SELF-ATTENTION MODEL - inspired from https://github.com/dlsyscourse/public_notebooks/blob/main/rnn_implementation.ipynb
The model has an initial linear layer (input_proj) that projects to the  hidden dimension (hidden_dim). 

This transformation above prepares the input for the self-attention mechanism, implemented via nn.MultiheadAttention. This attention mechanism allows the model to selectively focus on different parts of the time sequence, enabling it to capture dependencies and relationships within the sequence - here's a video that helped me understand it better.
* https://www.youtube.com/watch?v=IFKRf-BAqZo&feature=youtu.be

The multi-head attention layer also contains causal masking to ensure that the model cannot access future time steps to influence its decision

Output Layer:
The output_layer maps the transformed output to the target dimension (however many hours), predicting a single value per sample.

Training:
* MSE is used as the loss function
* MAE, MAPE, MSE are all calculated for each batch and averaged at the end

Validation:
* Evaluation occurs at the end of the epoch to see the validation loss

Testing:
* Average MSE, MAE, MAPE are all averaged

Graphs:
* Prediction vs. Actual. Sees how well the model performs. It compares the model predictions with actual target values
* training loss vs validation loss: sees if we're overfitting if the lines diverge. They don't




""""




