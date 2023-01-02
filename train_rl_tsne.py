import torch
import numpy as np
from utils import rl_tsne
from sklearn.datasets import load_digits
from torch.utils.tensorboard import SummaryWriter
from setting import my_args

torch.manual_seed(my_args.seed)
torch.cuda.manual_seed(my_args.seed)
torch.cuda.manual_seed_all(my_args.seed)

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

low_dim = rl_tsne(
    hidden_dim = my_args.hidden_dim, # hidden dim for agent
    data_label=digit_class,
    data=digits, # Data is mxn numpy array, each row a point
    n_components=my_args.n_components, # Number of dim to embed to
    perp=my_args.perplexity, # Perplexity (higher for more spread out data)
    perp_tol=my_args.perp_tol,
    steps=my_args.steps, # Iterations to run t-SNE for
    lr=my_args.lr, # Learning rate
    episodes=my_args.episodes,
    env_num=my_args.env_num,
    device_id=my_args.device_id,
    gamma=my_args.gamma, # Discounted factor
)