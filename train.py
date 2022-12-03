from simple_tsne import tsne, momentum_func
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits, digit_class = load_digits(return_X_y=True)
import numpy as np
digit_class = np.array([1,1,1])
digits = np.array([[1,1,1], [1,2,1], [1,1,3]])

low_dim = tsne(
    data=digits, # Data is mxn numpy array, each row a point
    data_lable=digit_class,
    n_components=2, # Number of dim to embed to
    perp=30, # Perplexity (higher for more spread out data)
    n_iter=500, # Iterations to run t-SNE for
    lr=100, # Learning rate
    momentum_fn=momentum_func, # Function returning momentum coefficient, this one is the default update schedule
    pbar=True, # Show progress bar
    random_state=42,# Seed for random initialization
    early_exaggeration =1,
)

# Plot results
plt.figure()
plt.scatter(low_dim[:,0], low_dim[:,1], c=digit_class)
plt.savefig("test.png")