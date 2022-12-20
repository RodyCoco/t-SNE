from RL_tsne import rl_tsne
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch
from linear import Network
import numpy as np

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

digits, digit_class = load_digits(return_X_y=True)
digit_class = digit_class[:50]
digits = digits[:50]

n_components = 2
model = Network(digits.shape[0]*n_components, digits.shape[0]*n_components)
model.double()

low_dim = rl_tsne(
    model=model,
    data_label=digit_class,
    data=digits, # Data is mxn numpy array, each row a point
    n_components=n_components, # Number of dim to embed to
    perp=5, # Perplexity (higher for more spread out data)
    n_iter=500, # Iterations to run t-SNE for
    lr=1e-3, # Learning rate
    pbar=True, # Show progress barv
    random_state=42, # Seed for random initialization
    split_num=1,
    # save_video=True,
)

fig, ax = plt.subplots()
scatter  = ax.scatter(low_dim[:,0], low_dim[:,1], c=digit_class, cmap=plt.get_cmap('turbo')) # cmap=plt.get_cmap('turbo'))
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
legend1 = ax.legend(*scatter.legend_elements(),
            loc="center left", title="Classes", bbox_to_anchor=(1, 0.5))
ax.add_artist(legend1)
plt.savefig("results/result.png")
plt.close()