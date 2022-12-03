from simple_tsne import tsne, momentum_func
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

digits, digit_class = load_digits(return_X_y=True)
import numpy as np
digits = np.array(
    [
        [1, -1, 1],
        [1.5, 0, 2],
        [2, 1, 0],
        [2.5, 0.5, 0.5]
    ]
)

digit_class = np.array([1,1,2,2])

low_dim = tsne(
    data=digits, # Data is mxn numpy array, each row a point
    data_label=digit_class,
    n_components=2, # Number of dim to embed to
    perp=2, # Perplexity (higher for more spread out data)
    n_iter=50, # Iterations to run t-SNE for
    lr=1, # Learning rate
    momentum_fn=momentum_func, # Function returning momentum coefficient, this one is the default update schedule
    pbar=True, # Show progress bar
    random_state=43, # Seed for random initialization
    early_exaggeration = 1
)
print(low_dim)
# Plot results
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.figure()
plt.scatter(low_dim[:,0], low_dim[:,1], c=digit_class)
plt.savefig("test.png")