import cv2
from utils import save_tsne_result, low_dim_affinities, squared_dist_mat, EPSILON, all_sym_affinities
from setting import my_args
import numpy as np
import torch
from policy import DiagonalGaussianPolicy
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import load_digits

np.random.seed(my_args.seed)
digits, digit_class = load_digits(return_X_y=True)
data_label = digit_class[:50]
data = digits[:50]

agent = DiagonalGaussianPolicy(N=data.shape[0], low_dim=my_args.n_components, \
    high_dim=data.shape[1], lr=my_args.lr, device_id=my_args.device_id, hidden_dim=my_args.hidden_dim)
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
    
    for t in range(my_args.steps):
        low_dim_data = Y.reshape(1, -1)
        high_dim_data = data.reshape(1, -1)
        s_t = torch.cat((low_dim_data, high_dim_data), dim=1).cuda(my_args.device_id)
        a_t = agent.act(s_t)
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