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
        mu = self.policy(s_t)
        var = torch.exp(self.policy.var)
        pi = torch.distributions.MultivariateNormal(mu, torch.diag(var))
        return pi