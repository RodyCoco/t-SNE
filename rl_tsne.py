from RL_tsne import rl_tsne, save_tsne_result, low_dim_affinities_3D, squared_dist_mat, EPSILON
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
    [1, 0, 0, 0 ,0],
    [0.9, 0, 0, 0 ,0],
    [0, 1, 0, 0 ,0],
    [0, 0.9, 0, 0 ,0],
    [0, 0, 1, 0 ,0],
    [0, 0, 0.9, 0 ,0],
    [0, 0, 0, 1 ,0],
    [0, 0, 0, 0.9 ,0],
    [0, 0, 0, 0 ,1],
    [0, 0, 0, 0 ,0.9],
]).astype(np.float64)

digit_class = np.array([1,1,2,2,3,3,4,4,5,5])

n_components = 2
steps = 50
device_id = 7

low_dim, P, agent = rl_tsne(
    hidden_dim = [20,10,5],
    data_label=digit_class,
    data=digits, # Data is mxn numpy array, each row a point
    n_components=n_components, # Number of dim to embed to
    perp=2, # Perplexity (higher for more spread out data)
    steps=50, # Iterations to run t-SNE for
    lr=2e-3, # Learning rate
    split_num=1,
    epochs=200,
    batch_size=64,
    save_video=True,
    device_id=device_id,
    gamma=0,
)