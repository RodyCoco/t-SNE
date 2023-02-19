import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
from policy import DiagonalGaussianPolicy, GNNPolicy
from torch_geometric.data import Data
EPSILON = 1e-12
diags_mask = None

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

def low_dim_affinities_3D(Y_dist_mat):
    bacth_size, data_size, _ = Y_dist_mat.shape
    numers = (1 + Y_dist_mat) ** (-1) # B,data_size,data_size
    diags = numers * diags_mask # B,data_size,data_size only diag
    denom =  numers.flatten(start_dim=1).sum(dim=1) - diags.flatten(start_dim=1).sum(dim=1) #B,1
    denom += EPSILON  # Avoid div/0
    denom = denom.reshape(bacth_size, 1, 1).repeat(1, data_size, data_size)
    Q = numers / denom
    Q = Q * (1-diags_mask)
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
    N=50
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
    
    init_mean = np.zeros(n_components)
    init_cov = np.identity(n_components) * 1e-4

    agent = GNNPolicy(
        var_dim = n_components*N,
        layer_num=3,
        input_dim=data.shape[-1]+n_components,
        hidden_dim=[200, 100],
        output_dim=n_components,
        lr=lr,
        device_id=device_id,
        )
    cur_time = str(datetime.now())[:-7]
    writer = SummaryWriter(log_dir="training_logs/"+cur_time)
    
    states_x = torch.empty((steps, env_num, N, data.shape[1]+n_components)).cuda(device_id).double()
    actions = torch.empty((steps, env_num, n_components*N)).cuda(device_id).double()
    rewards = torch.empty((steps, env_num, 1)).cuda(device_id).double()
    max_reward = -np.inf
    best_Y = None
    
    edge_index = torch.zeros((2, 2450)).long().cuda(device_id)
    
    global diags_mask
    diags_mask = torch.eye(N).repeat(env_num, 1, 1).cuda(device_id)
    
    for episode in range(episodes):
        index = np.arange(data.shape[0])
        np.random.shuffle(index)
        cur_data = data[index[:N]]
        Y =  np.random.uniform(-1, 1, size=(env_num, cur_data.shape[0], n_components))
        Y = torch.tensor(Y).cuda(device_id)
        origin_low_shape = Y.shape
        Y_dist_mat = squared_dist_mat(Y)
        Q = low_dim_affinities_3D(Y_dist_mat)
        Q = torch.clip(Q, EPSILON, None)
        P = all_sym_affinities(cur_data, perp, perp_tol)
        P = torch.clip(P, EPSILON, None).cuda(device_id)
        prev_KL_div = torch.sum(torch.sum(P * (torch.log(P) - torch.log(Q)), -1), -1)
        
        edge_attr = torch.zeros((2450, 1)).cuda(device_id)
        idx = -1
        for i in range(N):
            for j in range(N):
                if i == j: continue
                idx+=1
                edge_index[:,idx] = torch.tensor([i, j]).long()
                edge_attr[idx,:] = torch.tensor([P[i,j]]).double()
        cur_data = cur_data[None,:].repeat(env_num, 1, 1).cuda(device_id)
        
        for t in range(steps):
            
            # Define graph components
            x = torch.cat((cur_data, Y), dim=-1) # shape: 資料數量x(低維size+高維size)
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).cuda(device_id) 

            a_t = agent.act(graph)
            Y = a_t.reshape(origin_low_shape) + Y
            
            Y_dist_mat = squared_dist_mat(Y)
            Q = low_dim_affinities_3D(Y_dist_mat).cuda(device_id)
            Q = torch.clip(Q, EPSILON, None)

            cur_KL_div = torch.sum(torch.sum(P * (torch.log(P) - torch.log(Q)), -1), -1)
            r_t = prev_KL_div - cur_KL_div
            states_x[t] = graph.x
            actions[t] = a_t
            r_t.requires_grad = False
            rewards[t] = r_t.reshape(env_num, 1)
            prev_KL_div = cur_KL_div

        returns = calculate_returns(rewards, gamma=gamma).cuda(device_id)
        # reward normalization
        returns = (returns - torch.mean(returns)) / (torch.std(returns) + 1e-10)
        agent.learn((states_x, edge_index, edge_attr), actions, returns)
        total_reward = torch.sum(rewards)
        writer.add_scalar("mean total reward", total_reward/env_num, episode + 1)
        print(f"Episode[{episode+1}/{episodes}], mean total reward:{total_reward/env_num:5f}")
        if total_reward > max_reward:
            max_reward = total_reward
            best_Y = Y
            torch.save(agent.policy.state_dict(), "model.pkl")
            print("save model")

    return best_Y.detach().cpu().numpy()

def save_tsne_result(low_dim, digit_class, fig_dir):
    fig, ax = plt.subplots()
    scatter  = ax.scatter(low_dim[:,0], low_dim[:,1], c=digit_class, cmap=plt.get_cmap('turbo'))
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    legend1 = ax.legend(*scatter.legend_elements(),
                loc="center left", title="Classes", bbox_to_anchor=(1, 0.5))
    ax.add_artist(legend1)
    plt.savefig(fig_dir)
    plt.close()