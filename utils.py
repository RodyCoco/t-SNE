import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
from policy import DiagonalGaussianPolicy
EPSILON = 1e-12

def squared_dist_mat(X):
    
    D = torch.square(torch.cdist(X, X))
    return D


def pairwise_affinities(data, sigmas, dist_mat):
    
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

    asym_affinities = torch.clip(
        asym_affinities, EPSILON, None
    )  # Some are so small that log2 fails.
    out = -torch.sum(asym_affinities * torch.log2(asym_affinities), dim=1)
    
    return out


def get_perplexities(asym_affinities):
    return 2 ** get_entropies(asym_affinities)


def all_sym_affinities(data, perp, tol, attempts=100):
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
    numers = (1 + Y_dist_mat) ** (-1)
    denom = torch.sum(numers) - torch.sum(torch.diag(numers))
    denom += EPSILON  # Avoid div/0
    Q = numers / denom
    Q = Q.fill_diagonal_(0.0)
    return Q[None,:,:]

def low_dim_affinities_3D( Y_dist_mat):
    Q = list(map(low_dim_affinities, Y_dist_mat))
    Q = torch.cat(Q)
    return Q


def calculate_returns(rewards, gamma):
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
    steps,
    lr,
    perp_tol=1e-8,
    seed=0,
    gamma=0,
    episodes=1,
    device_id=9,
    env_num=8,
):  
    video_dir = "plots/video"
    figure_dir = "plots/figure"
    path_list = [video_dir, figure_dir]
    for path in path_list:
        if not os.path.isdir(path):
            os.makedirs(path)

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
    writer = SummaryWriter(log_dir="training_logs/"+cur_time)
    
    low_size = n_components*data.shape[0]
    high_size = data.reshape(-1).shape[0]
    states = torch.empty((steps, env_num, low_size+high_size)).cuda(device_id).double()
    actions = torch.empty((steps, env_num, low_size)).cuda(device_id).double()
    rewards = torch.empty((steps, env_num, 1)).cuda(device_id).double()
    max_reward = -np.inf
    best_Y = None
    
    for episode in range(episodes):
        
        # Y = np.random.multivariate_normal(mean=init_mean, cov=init_cov, size=(env_num, data.shape[0]))
        Y =  np.random.uniform(-1, 1, size=(env_num, data.shape[0], n_components))
        Y = torch.tensor(Y).cuda(device_id)
        origin_low_shape = Y.shape
        Y_dist_mat = squared_dist_mat(Y)
        Q = low_dim_affinities_3D(Y_dist_mat)
        Q = torch.clip(Q, EPSILON, None)
        prev_KL_div = torch.sum(torch.sum(P * (torch.log(P) - torch.log(Q)), -1), -1)
        
        for t in range(steps):
            low_dim_data = Y.reshape(env_num, -1)
            high_dim_data = data.reshape(1, -1).repeat(env_num, 1)
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
            rewards[t] = r_t.reshape(env_num, 1)
            prev_KL_div = cur_KL_div
            
        returns = calculate_returns(rewards, gamma=gamma).cuda(device_id)
        # reward normalization
        returns = (returns - torch.mean(returns)) / (torch.std(returns) + 1e-10)
        agent.learn(states, actions, returns)
        total_reward = torch.sum(rewards)
        writer.add_scalar("mean total reward", total_reward/env_num, episode + 1)
        print(f"Episode[{episode}/{episodes}], mean total reward:{total_reward/env_num:5f}")
        if total_reward > max_reward:
            max_reward = total_reward
            best_Y = Y
            torch.save(agent.policy.state_dict(), "model.pkl")
            print("save model")
            
    return best_Y.detach().cpu().numpy()

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