from simple_tsne import tsne, momentum_func, split_tsne
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

digits, digit_class = load_digits(return_X_y=True)
digit_class = digit_class[:500]
digits = digits[:500]

low_dim = split_tsne(
    data_label=digit_class,
    data=digits, # Data is mxn numpy array, each row a point
    n_components=2, # Number of dim to embed to
    perp=30, # Perplexity (higher for more spread out data)
    n_iter=500, # Iterations to run t-SNE for
    lr=20, # Learning rate
    # momentum_fn=momentum_func, # Function returning momentum coefficient, this one is the default update schedule
    pbar=True, # Show progress barv
    random_state=42, # Seed for random initialization
    split_num=4,
    # save_video = True
    # exp_decay = 0.995
)

fig, ax = plt.subplots()
scatter  = ax.scatter(low_dim[:,0], low_dim[:,1], c=digit_class, cmap=plt.get_cmap('turbo'))
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
legend1 = ax.legend(*scatter.legend_elements(),
            loc="center left", title="Classes", bbox_to_anchor=(1, 0.5))
ax.add_artist(legend1)
plt.savefig("results/result.png")
plt.close()