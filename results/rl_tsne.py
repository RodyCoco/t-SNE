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
digit_class = digit_class[:500]
digits = digits[:500]

digits = np.array([
    [1,-1,1],
    [1.5,0,2],
    [2,1,0],
    [2.5,0.5,0.5],
])
digit_class = np.array([1,1,2,2])

model = Network(digits.shape[0]*2, digits.shape[0]*2, hidden_dim=[5,4,3])
model.double()

low_dim = rl_tsne(
    model=model,
    data_label=digit_class,
    data=digits, # Data is mxn numpy array, each row a point
    n_components=2, # Number of dim to embed to
    perp=2, # Perplexity (higher for more spread out data)
    n_iter=50, # Iterations to run t-SNE for
    lr=0.01, # Learning rate
    after_lr = 5e-3,
    pbar=True, # Show progress barv
    random_state=42, # Seed for random initialization
    split_num=1,
    early_exaggeration=1.0,
    save_video=True,
)

fig, ax = plt.subplots()
scatter  = ax.scatter(low_dim[:,0], low_dim[:,1], c=digit_class,) # cmap=plt.get_cmap('turbo'))
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
legend1 = ax.legend(*scatter.legend_elements(),
            loc="center left", title="Classes", bbox_to_anchor=(1, 0.5))
ax.add_artist(legend1)
plt.savefig("results/result.png")
plt.close()