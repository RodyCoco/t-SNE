"""
implementation of van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.
"""

import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
EPSILON = 1e-12
gpu_number = 4


def squared_dist_mat(X):
    """calculates the squared eucledian distance matrix

    function source: https://lvdmaaten.github.io/tsne/
    Parameters:
    X : ndarray of shape (n_samples, n_features)

    Returns:
    D: Squared eucledian distance matrix of shape (n_samples, n_samples)

    """
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
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
    numers = np.exp(inner)
    denoms = np.sum(numers, axis=1) - np.diag(numers) #不採計自己的距離
    denoms = denoms.reshape(-1, 1)
    denoms += EPSILON  # Avoid div/0
    P = numers / denoms
    np.fill_diagonal(P, 0.0)
    return P


def get_entropies(asym_affinities):
    """
    Row-wise Shannon entropy of pairwise affinity matrix P

    Parameters:
    asym_affinities: pairwise affinity matrix of shape (n_samples, n_samples)

    Returns:
    array-like row-wise Shannon entropy of shape (n_samples,)
    """
    asym_affinities = np.clip(
        asym_affinities, EPSILON, None
    )  # Some are so small that log2 fails.
    return -np.sum(asym_affinities * np.log2(asym_affinities), axis=1)


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
    dist_mat = squared_dist_mat(data)  # mxm

    sigma_maxs = np.full(data.shape[0], 1e12)

    # zero here causes div/0, /2sigma**2 in P calc
    sigma_mins = np.full(data.shape[0], 1e-12)

    current_perps = np.full(data.shape[0], np.inf)

    while (not np.allclose(current_perps, perp, atol=tol)) and attempts > 0:
        sigmas = (sigma_mins + sigma_maxs) / 2
        P = pairwise_affinities(data, sigmas.reshape(-1, 1), dist_mat)
        current_perps = get_perplexities(P)
        attempts -= 1
        sigma_maxs = np.where(current_perps>perp, sigmas, sigma_maxs)
        sigma_mins = np.where(current_perps<perp, sigmas, sigma_mins)
        
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

    numers = 1/(1 + Y_dist_mat) if torch.is_tensor(Y_dist_mat) else torch.tensor(1/(1 + Y_dist_mat)).cuda(gpu_number)
    denom = torch.sum(numers) - torch.sum(torch.diag(numers))
    denom += EPSILON  # Avoid div/0
    Q = numers / denom
    B = torch.eye(Q.shape[0])
    B = torch.where(B==1,0,1).cuda(gpu_number)
    Q = Q*B
    return Q


def compute_grad(cur_P, cur_Y, model, prev_kl_div, optimizer):
    torch.autograd.set_detect_anomaly(True)
    origin_shape = cur_Y.shape
    cur_Y = torch.tensor(cur_Y.reshape(1, 1, -1)).cuda(gpu_number)
    action = model(cur_Y)
    
    action = action.reshape(origin_shape)
    cur_Y = cur_Y.reshape(origin_shape)
    cur_Y = action # cur_Y + action
    
    Y_dist_mat = torch.cdist(cur_Y, cur_Y, p=2).pow(2)
    cur_Q = low_dim_affinities(cur_Y, Y_dist_mat)
    cur_Q = torch.clip(cur_Q, EPSILON, None).cuda(gpu_number)
    
    # KL div
    cur_P = torch.tensor(cur_P).cuda(gpu_number)
    cur_kl_div = torch.sum(cur_P * (torch.log(cur_P) - torch.log(cur_Q)))
    optimizer.zero_grad()
    loss = cur_kl_div
    loss.backward()
    optimizer.step()
    
    return action


def momentum_func(t):
    """returns optimization parameter

    Parameters:
    t: integer, iteration number

    Returns:
    float representing the momentum term added to the gradient
    """
    if t < 250:
        return 0.5
    else:
        return 0.8

def rl_tsne(
    model,
    data,
    data_label,
    n_components,
    perp,
    n_iter,
    lr,
    after_lr,
    perp_tol=1e-8,
    early_exaggeration=4.0,
    pbar=False,
    random_state=None,
    split_num=1,
    seed=0,
    save_video=False,
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
    model = model.cuda(gpu_number)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    np.random.seed(seed)
    rand = np.random.RandomState(random_state)
    
    P = all_sym_affinities(data, perp, perp_tol) * early_exaggeration
    P = np.clip(P, EPSILON, None)

    init_mean = np.zeros(n_components)
    init_cov = np.identity(n_components) * 1e-4

    Y = rand.multivariate_normal(mean=init_mean, cov=init_cov, size=data.shape[0])

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
        # np.random.shuffle(data_order)
        for i in range(split_num):
            start_idx, end_idx = i*split_data_size, (i+1)*split_data_size
            cur_Y = Y[data_order[start_idx:end_idx]]
            cur_P = P[data_order[start_idx:end_idx]]
            cur_P = cur_P[:, data_order[start_idx:end_idx]]

            Y_dist_mat = squared_dist_mat(Y)
            Q = low_dim_affinities(Y, Y_dist_mat)
            Q = np.clip(Q.detach().cpu().numpy(), EPSILON, None)
            prev_kl_div = np.sum(P * (np.log(P) - np.log(Q)))
            
            action = compute_grad(cur_P, cur_Y.copy(), model, prev_kl_div, optimizer)
            Y[data_order[start_idx:end_idx]] = action.detach().cpu().numpy() # cur_Y + action.detach().cpu().numpy()
        
        Y_dist_mat = squared_dist_mat(Y)
        Q = low_dim_affinities(Y, Y_dist_mat)
        Q = np.clip(Q.detach().cpu().numpy(), EPSILON, None)
        loss = np.sum(P * (np.log(P) - np.log(Q)))
        writer.add_scalar("Loss", loss, t + 1)
        if loss < min_loss:
            min_loss = loss
        if t == 100:
            P = P / early_exaggeration
            optimizer = torch.optim.Adam(model.parameters(), lr=after_lr)

        if save_video:
            fig, ax = plt.subplots()
            plt.ylim(-2, 6)
            plt.xlim(-3, 10)
            scatter  = ax.scatter(Y[:,0], Y[:,1], c=data_label, )# cmap=plt.get_cmap('turbo'))
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