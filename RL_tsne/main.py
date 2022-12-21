"""
implementation of van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.
"""
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
EPSILON = 1e-12


def squared_dist_mat(X):
    """calculates the squared eucledian distance matrix

    function source: https://lvdmaaten.github.io/tsne/
    Parameters:
    X : ndarray of shape (n_samples, n_features)

    Returns:
    D: Squared eucledian distance matrix of shape (n_samples, n_samples)

    """
    sum_X = torch.sum(torch.square(X), 1)
    D = torch.transpose(-2 * torch.mm(X, torch.transpose(X,0,1)) + sum_X, 0, 1) + sum_X
    return D


def pairwise_affinities(data, sigmas, dist_mat):
    """calculates the pairwise affinities p_{j|i} using the given values of sigma

    Parameters:
    data : ndarray of shape (n_samples, n_features)
    sigmas : column array of shape (n_samples, 1)
    dist_mat : data distance matrix; ndarray of shape (n_samples, n_samples)

    Returns:
    P: pairwise affinity matrix of size (n_samples, n_samples)

    """
    
    assert sigmas.shape == (data.shape[0], 1)

    inner = (-dist_mat) / (2 * (sigmas ** 2))
    numers = torch.exp(inner)
    denoms = torch.sum(numers, dim=1) - torch.diag(numers) #不採計自己的距離
    denoms = denoms.reshape(-1, 1)
    denoms += EPSILON  # Avoid div/0
    P = numers / denoms
    P = P.fill_diagonal_(0.0)   

    return P


def get_entropies(asym_affinities):
    """
    Row-wise Shannon entropy of pairwise affinity matrix P

    Parameters:
    asym_affinities: pairwise affinity matrix of shape (n_samples, n_samples)

    Returns:
    array-like row-wise Shannon entropy of shape (n_samples,)
    """
    
    asym_affinities = torch.clip(
        asym_affinities, EPSILON, None
    )  # Some are so small that log2 fails.
    out = -torch.sum(asym_affinities * torch.log2(asym_affinities), dim=1)
    
    return out


def get_perplexities(asym_affinities):
    """
    compute perplexities of pairwise affinity matrix P

    Parameters:
    asym_affinities: pairwise affinity matrix of shape (n_samples, n_samples)

    Returns:
    array-like row-wise perplexities of shape (n_samples,)
    """
    return 2 ** get_entropies(asym_affinities)


def all_sym_affinities(data, perp, tol, attempts=100):
    """
    finds the data specific sigma values and calculates the symmetric affinities matrix P
    Parameters:
    data : ndarray of shape (n_samples, n_features)
    perp : float, cost function parameter
    tol : float, tolerance of how close the current perplexity is to the target perplexity
    attempts : int, a maximum limit to the binary search attempts

    Returns:
    P: Symmetric affinities matrix of shape (n_samples, n_samples)

    """
    perp = torch.tensor([perp], dtype=torch.float64)
    
    dist_mat = squared_dist_mat(data)  # mxm

    sigma_maxs = torch.full((data.shape[0],), 1e12)

    # zero here causes div/0, /2sigma**2 in P calc
    sigma_mins = torch.full((data.shape[0],), 1e-12)

    current_perps = torch.full((data.shape[0],), np.inf, dtype=torch.float64) 

    while (not torch.allclose(current_perps, perp, atol=tol)) and attempts > 0:
        sigmas = (sigma_mins + sigma_maxs) / 2
        P = pairwise_affinities(data, sigmas.reshape(-1, 1), dist_mat)
        current_perps = get_perplexities(P)
        attempts -= 1
        sigma_maxs = torch.where(current_perps>perp, sigmas, sigma_maxs)
        sigma_mins = torch.where(current_perps<perp, sigmas, sigma_mins)
        
    if attempts == 0:
        print(
            "Warning: Ran out attempts before converging, try a different perplexity?"
        )
    P = (P + P.T) / (2 * data.shape[0])
    return P


def low_dim_affinities(Y, Y_dist_mat):
    """
    computes the low dimensional affinities matrix Q
    Parameters:
    Y : low dimensional representation of the data, ndarray of shape (n_samples, n_components)
    Y_dist_mat : Y distance matrix; ndarray of shape (n_samples, n_samples)

    Returns:
    Q: Symmetric low dimensional affinities matrix of shape (n_samples, n_samples)

    """
    numers = (1 + Y_dist_mat) ** (-1)
    denom = torch.sum(numers) - torch.sum(torch.diag(numers))
    denom += EPSILON  # Avoid div/0
    Q = numers / denom
    Q = Q.fill_diagonal_(0.0)
    return Q


def compute_grad(P, Q, Y, Y_dist_mat):
    """
    computes the gradient vector needed to update the Y values
    Parameters:
    P: Symmetric affinities matrix of shape (n_samples, n_samples)
    Q: Symmetric low dimensional affinities matrix of shape (n_samples, n_samples)
    Y : low dimensional representation of the data, ndarray of shape (n_samples, n_components)
    Y_dist_mat : Y distance matrix; ndarray of shape (n_samples, n_samples)

    Returns:
    grad: the gradient vector, shape (n_samples, n_components)

    """
    Ydiff = Y[:, None, :] - Y[None, :, :]
    pq_factor = (P - Q)[:, :, None]
    dist_factor = ((1 + Y_dist_mat) ** (-1))[:, :, None]
    return torch.sum(4 * pq_factor * Ydiff * dist_factor, axis=1)

def rl_tsne(
    data,
    data_label,
    n_components,
    perp,
    n_iter,
    lr,
    perp_tol=1e-8,
    early_exaggeration=4.0,
    pbar=False,
    random_state=None,
    split_num=1,
    seed=0,
    save_video=False,
    exp_decay=1,
):
    """calculates the pairwise affinities p_{j|i} using the given values of sigma

    Parameters:
    data : ndarray of shape (n_samples, n_features)
    n_components : int, target number of dimensions
    perp : float, cost function parameter
    n_iter : number of iterations to run, recommended to be no less than 250
    lr : learning rate
    momentum_fn : function that controls the momentum term
    perp_tol : float, tolerance of how close the current perplexity is to the target perplexity
    early_exaggeration : optimization parameter
    pbar : flag to show tqdm progress bar during iterations
    random_state : determines the random number generator, set for reproducible results


    Returns:
    Y: low dimensional representation of the data, ndarray of shape (n_samples, n_components)

    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    data = torch.tensor(data)
    P = all_sym_affinities(data, perp, perp_tol) * early_exaggeration
    P = torch.clip(P, EPSILON, None)

    init_mean = np.zeros(n_components)
    init_cov = np.identity(n_components) * 1e-4

    Y = np.random.multivariate_normal(mean=init_mean, cov=init_cov, size=data.shape[0])
    Y = torch.tensor(Y)
    
    iter_range = range(n_iter)
    if pbar:
        iter_range = tqdm(iter_range, "Iterations")
    if save_video:
        videowrite = cv2.VideoWriter('results/result.mp4',cv2.VideoWriter_fourcc(*'mp4v'),20,(640,480))
    
    cur_time = str(datetime.now())[:-7]
    writer = SummaryWriter(log_dir="logs/"+cur_time)
    
    data_size = data.shape[0]
    split_data_size = int(np.ceil(data_size/split_num))
    min_loss = np.inf
    
    for t in iter_range:
        data_order = np.arange(data_size)
        np.random.shuffle(data_order)
        for i in range(split_num):
            start_idx, end_idx = i*split_data_size, (i+1)*split_data_size
            cur_Y = Y[data_order[start_idx:end_idx]]
            cur_P = P[data_order[start_idx:end_idx]]
            cur_P = cur_P[:, data_order[start_idx:end_idx]]

            Y_dist_mat = squared_dist_mat(cur_Y)
            Q = low_dim_affinities(cur_Y, Y_dist_mat)
            Q = torch.clip(Q, EPSILON, None)

            grad = compute_grad(cur_P, Q, cur_Y, Y_dist_mat)*((exp_decay)**(t-500))
            Y[data_order[start_idx:end_idx]] = cur_Y - lr * grad
        
        Y_dist_mat = squared_dist_mat(Y)
        Q = low_dim_affinities(Y, Y_dist_mat)
        Q = torch.clip(Q, EPSILON, None)
        loss = torch.sum(P * (torch.log(P) - torch.log(Q)))
        writer.add_scalar("Loss", loss, t + 1)
        if loss < min_loss:
            min_loss = loss
        if t == 100:
            P = P / early_exaggeration

        if save_video:
            fig, ax = plt.subplots()
            scatter  = ax.scatter(Y[:,0], Y[:,1], c=data_label, cmap=plt.get_cmap('turbo'))
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
            legend1 = ax.legend(*scatter.legend_elements(),
                        loc="center left", title="Classes", bbox_to_anchor=(1, 0.5))
            ax.add_artist(legend1)
            plt.savefig("results/t.png")
            img = cv2.imread("results/t.png")
            videowrite.write(img)
            plt.close()
    print("min_loss:", min_loss)
    return Y