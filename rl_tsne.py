from RL_tsne import rl_tsne, save_tsne_result
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

digits = np.array([
    [1,1,1],
    [1,1,2],
    [3,4,7],
    [3,4,8],
], dtype=np.float64) 

digit_class = np.array([1,1,2,2])

n_components = 2

low_dim = rl_tsne(
    data_label=digit_class,
    data=digits, # Data is mxn numpy array, each row a point
    n_components=n_components, # Number of dim to embed to
    perp=1, # Perplexity (higher for more spread out data)
    n_iter=50, # Iterations to run t-SNE for
    lr=5e-3, # Learning rate
    split_num=1,
    epochs=200,
    batch_size=16,
    save_video=True,
)

save_tsne_result(low_dim[0], digit_class, "results/result.png")

