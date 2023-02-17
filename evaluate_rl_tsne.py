import cv2
from utils import save_tsne_result, low_dim_affinities, squared_dist_mat, EPSILON, all_sym_affinities
from setting import my_args
import numpy as np
import torch
import os
from policy import DiagonalGaussianPolicy, GNNPolicy
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import load_digits
from torch_geometric.data import Data

video_dir = "plots/video"
figure_dir = "plots/figure"
path_list = [video_dir, figure_dir]
for path in path_list:
    if not os.path.isdir(path):
        os.makedirs(path)
        
np.random.seed(my_args.seed)
digits, digit_class = load_digits(return_X_y=True)
data_label = digit_class[:50]
data = digits[:50]

agent = GNNPolicy(
        var_dim = my_args.n_components*data.shape[0],
        layer_num=3,
        input_dim=data.shape[-1]+my_args.n_components,
        hidden_dim=[200, 100],
        output_dim=my_args.n_components,
        lr=my_args.lr,
        device_id=my_args.device_id,
        )
agent.policy.load_state_dict(torch.load("model.pkl"))
video_per_frame = 10


if __name__ == "__main__":
    videowrite = cv2.VideoWriter('plots/video/result.mp4',cv2.VideoWriter_fourcc(*'mp4v'),video_per_frame,(640,480))
    init_mean = np.zeros(my_args.n_components)
    init_cov = np.identity(my_args.n_components) * 1e-4
    # Y = np.random.multivariate_normal(mean=init_mean, cov=init_cov, size=data.shape[0])
    Y = np.random.uniform(-1, 1, size=(data.shape[0], my_args.n_components))
    Y = torch.tensor(Y).cuda(my_args.device_id)
    data = torch.tensor(data)
    P = all_sym_affinities(data, my_args.perplexity, my_args.perp_tol)
    P = torch.clip(P, EPSILON, None).cuda(my_args.device_id)
    edge_index = torch.zeros((2, 2450)).long().cuda(my_args.device_id)
    edge_attr = torch.zeros((2450, 1)).cuda(my_args.device_id)
    idx = -1
    for i in range(50):
        for j in range(50):
            if i == j: continue
            idx+=1
            edge_index[:,idx] = torch.tensor([i, j]).long()
            edge_attr[idx,:] = torch.tensor([P[i,j]]).double()
    data = data.cuda(my_args.device_id)
    origin_low_shape = Y.shape
    cur_time = str(datetime.now())[:-7]
    writer = SummaryWriter(log_dir="evaluating_logs/"+cur_time)
    
    save_tsne_result(Y.detach().cpu().numpy(), data_label, "plots/figure/t.png")
    img = cv2.imread("plots/figure/t.png")
    videowrite.write(img)
    
    Y_dist_mat = squared_dist_mat(Y)
    Q = low_dim_affinities(Y_dist_mat)
    Q = torch.clip(Q, EPSILON, None)
    KL_div = torch.sum(torch.sum(P * (torch.log(P) - torch.log(Q)), -1), -1)
    print(f"steps[0/{my_args.steps}], KL_div: {KL_div.item():5f}")
    from itertools import permutations
    edge_index = torch.tensor(list(permutations(np.arange(data.shape[0]),2))) # shape: 2*[(資料數量(資料數量-1))]
    edge_index = torch.permute(edge_index, (1, 0)).cuda(my_args.device_id)
    
    for t in range(my_args.steps):
        # Define graph components
        x = torch.cat((data, Y), dim=-1) # shape: 資料數量x(低維size+高維size)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).cuda(my_args.device_id) 
        a_t = agent.act(graph)
        Y = a_t.reshape(origin_low_shape) + Y
        save_tsne_result(Y.detach().cpu().numpy(), data_label, "plots/figure/t.png")
        img = cv2.imread("plots/figure/t.png")
        videowrite.write(img)
        Y_dist_mat = squared_dist_mat(Y)
        Q = low_dim_affinities(Y_dist_mat)
        Q = torch.clip(Q, EPSILON, None)
        KL_div = torch.sum(torch.sum(P * (torch.log(P) - torch.log(Q)), -1), -1)
        writer.add_scalar("KL div", KL_div, t + 1)
        print(f"steps[{t+1}/{my_args.steps}], KL_div: {KL_div.item():5f}")