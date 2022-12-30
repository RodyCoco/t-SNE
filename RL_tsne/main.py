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
from policy import DiagonalGaussianPolicy
EPSILON = 1e-12

def squared_dist_mat(X):
    """calculates the squared eucledian distance matrix

    function source: https://lvdmaaten.github.io/tsne/
    Parameters:
    X : ndarray of shape (n_samples, n_features)

    Returns:
    D: Squared eucledian distance matrix of shape (n_samples, n_samples)

    """
    D = torch.square(torch.cdist(X, X))
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


def low_dim_affinities(Y_dist_mat):
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
    return Q[None,:,:]

def low_dim_affinities_3D( Y_dist_mat):
    """
    computes the low dimensional affinities matrix Q
    Parameters:
    Y : low dimensional representation of the data, ndarray of shape (n_samples, n_components)
    Y_dist_mat : Y distance matrix; ndarray of shape (n_samples, n_samples)

    Returns:
    Q: Symmetric low dimensional affinities matrix of shape (n_samples, n_samples)

    """
    Q = list(map(low_dim_affinities, Y_dist_mat))
    Q = torch.cat(Q)
    return Q


def calculate_returns(rewards, gamma=0.9):
    result = torch.empty_like(rewards)
    result[-1] = rewards[-1]
    for t in range(len(rewards)-2, -1, -1):
        result[t] = rewards[t] + gamma*result[t+1]
    return result

def rl_tsne(
    hidden_dim,
    data,
    data_label,
    n_components,
    perp,
    n_iter,
    lr,
    perp_tol=1e-8,
    split_num=1,
    seed=0,
    gamma=0,
    save_video=False,
    epochs=1,
    device_id=9,
    batch_size=8,
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
    # Set all tensors to the first CUDA device

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Y: 低維資料
    # P: 高維資料 symmetric affinities matrix
    # Q: 低維資料 symmetric affinities matrix
    
    data = torch.tensor(data)
    P = all_sym_affinities(data, perp, perp_tol)
    P = torch.clip(P, EPSILON, None).cuda(device_id)
    data = data.cuda(device_id)
    
    init_mean = np.zeros(n_components)
    init_cov = np.identity(n_components) * 1e-4

    agent = DiagonalGaussianPolicy(N=data.shape[0], low_dim=n_components, high_dim=data.shape[1], lr=lr, device_id=device_id, hidden_dim=hidden_dim)
    cur_time = str(datetime.now())[:-7]
    writer = SummaryWriter(log_dir="logs/"+cur_time)
    
    low_size = n_components*data.shape[0]
    high_size = data.reshape(-1).shape[0]
    states = torch.empty((n_iter, batch_size, low_size+high_size)).cuda(device_id).double()
    actions = torch.empty((n_iter, batch_size, low_size)).cuda(device_id).double()
    rewards = torch.empty((n_iter, batch_size, 1)).cuda(device_id).double()
    max_reward = -np.inf
    best_Y, best_init_Y = None, None
    
    for epoch in range(epochs):
        
        Y = np.random.multivariate_normal(mean=init_mean, cov=init_cov, size=(batch_size, data.shape[0]))
        init_Y = Y = torch.tensor(Y).cuda(device_id)
        origin_low_shape = Y.shape
        Y_dist_mat = squared_dist_mat(Y)
        Q = low_dim_affinities_3D(Y_dist_mat)
        Q = torch.clip(Q, EPSILON, None)
        prev_KL_div = torch.sum(torch.sum(P * (torch.log(P) - torch.log(Q)), -1), -1)
        for t in range(n_iter):
            low_dim_data = Y.reshape(batch_size, -1)
            high_dim_data = data.reshape(1, -1).repeat(batch_size, 1)
            s_t = torch.cat((low_dim_data, high_dim_data), dim=-1)
            
            a_t = agent.act(s_t)
            Y = a_t.reshape(origin_low_shape) + Y
            
            Y_dist_mat = squared_dist_mat(Y)
            Q = low_dim_affinities_3D(Y_dist_mat).cuda(device_id)
            Q = torch.clip(Q, EPSILON, None)

            cur_KL_div = torch.sum(torch.sum(P * (torch.log(P) - torch.log(Q)), -1), -1)
            r_t = prev_KL_div - cur_KL_div
            states[t] = s_t
            actions[t] = a_t
            r_t.requires_grad = False
            rewards[t] = r_t.reshape(batch_size, 1)
    
            prev_KL_div = cur_KL_div
        
        returns = calculate_returns(rewards, gamma=gamma).cuda(device_id)
        agent.learn(states, actions, returns)
        total_reward = torch.sum(rewards)
        writer.add_scalar("mean total reward", total_reward/batch_size, epoch + 1)
        print(f"epoch:{epoch}, mean total reward:{total_reward/batch_size}")
        if total_reward > max_reward:
            max_reward = total_reward
            import copy
            best_Y, best_init_Y, best_agent = Y, init_Y, copy.deepcopy(agent)
            torch.save(agent.policy.state_dict(), "model.pkl")
            print("save model")
            
    if save_video:
        videowrite = cv2.VideoWriter('results/result.mp4',cv2.VideoWriter_fourcc(*'mp4v'),20,(640,480))
        Y = np.random.multivariate_normal(mean=init_mean, cov=init_cov, size=data.shape[0])
        Y = torch.tensor(Y).cuda(device_id)
        origin_low_shape = Y.shape
        for t in range(n_iter):
            low_dim_data = Y.reshape(1, -1)
            high_dim_data = data.reshape(1, -1)
            s_t = torch.cat((low_dim_data, high_dim_data), dim=1).cuda(device_id)
            a_t = agent.act(s_t)
            Y = a_t.reshape(origin_low_shape) + Y
            save_tsne_result(Y.detach().cpu().numpy(), data_label, "results/t.png")
            img = cv2.imread("results/t.png")
            videowrite.write(img)
            
    return best_Y.detach().cpu().numpy(), P, best_agent

def save_tsne_result(low_dim, digit_class, fig_dir):
    fig, ax = plt.subplots()
    scatter  = ax.scatter(low_dim[:,0], low_dim[:,1], c=digit_class,)# cmap=plt.get_cmap('turbo'))
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    legend1 = ax.legend(*scatter.legend_elements(),
                loc="center left", title="Classes", bbox_to_anchor=(1, 0.5))
    ax.add_artist(legend1)
    plt.savefig(fig_dir)
    plt.close()