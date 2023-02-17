import torch

class Policy:
    def pi(self, s_t):
        '''
        returns the probability distribution over actions
        (torch.distributions.Distribution)

        s_t (np.ndarray): the current state
        '''
        raise NotImplementedError

    def act(self, s_t):
        '''
        s_t (np.ndarray): the current state
        Because of environment vectorization, this will produce
        E actions where E is the number of parallel environments.
        '''
        a_t = self.pi(s_t).sample()
        return a_t

    def learn(self, states, actions, returns):
        '''
        states (np.ndarray): the list of states encountered during
                             rollout
        actions (np.ndarray): the list of actions encountered during
                              rollout
        returns (np.ndarray): the list of returns encountered during
                              rollout

        Because of environment vectorization, each of these has first
        two dimensions TxE where T is the number of time steps in the
        rollout and E is the number of parallel environments.
        '''
        log_prob = self.pi(states).log_prob(actions)
        loss = torch.mean(-log_prob*returns[:,:,0])

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

class DiagonalGaussianPolicy(Policy):
    def __init__(self, N, low_dim, high_dim, hidden_dim=[10,8,6],  lr=1e-4, device_id=0):
        '''
        env (gym.Env): the environment
        lr (float): learning rate
        '''
        from linear import Network
        self.policy = Network(N, low_dim, high_dim, hidden_dim).double().cuda(device_id)
        self.opt = torch.optim.Adam(list(self.policy.parameters()), lr=lr)

    def pi(self, s_t):
        '''
        returns the probability distribution over actions
        s_t (np.ndarray): the current state
        '''
        if type(s_t) == tuple:
            mu = self.policy(s_t[0], s_t[1])
        else:
            mu = self.policy(s_t)
        var = torch.exp(self.policy.var)
        pi = torch.distributions.MultivariateNormal(mu, torch.diag(var))
        return pi
    
class GNNPolicy(Policy):
    def __init__(self, var_dim, layer_num, input_dim, hidden_dim, output_dim,  lr=1e-4, device_id=0):
        '''
        env (gym.Env): the environment
        lr (float): learning rate
        '''
        from GCN import GCN

        self.policy = GCN(
            var_dim=var_dim, 
            input_dim=input_dim, 
            hidden_dim=hidden_dim,
            output_dim=output_dim
            ).double().cuda(device_id)
        pytorch_total_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        input(pytorch_total_params)
        self.opt = torch.optim.Adam(list(self.policy.parameters()), lr=lr)

    def pi(self, s_t):
        '''
        returns the probability distribution over actions
        s_t (np.ndarray): the current state
        '''
        if type(s_t) == tuple:
            mu = self.policy(s_t[0], s_t[1], s_t[2])
        else:
            mu = self.policy(s_t.x, s_t.edge_index, s_t.edge_attr)
        var = torch.exp(self.policy.var)
        if len(mu.shape) == 3:
            batch_size, _, _ = mu.shape
            mu = mu.reshape(batch_size, -1)
        elif len(mu.shape) == 2:
            mu = mu.reshape(-1)
        else:
            steps_size, batch_size, _, _ = mu.shape
            mu = mu.reshape(steps_size, batch_size, -1)
        pi = torch.distributions.MultivariateNormal(mu, torch.diag(var))
        return pi