from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data

# torch.random.manual_seed(31)
torch.set_default_dtype(torch.float64)

T = 10

# Fetch GPU
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Found GPU.")

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

class PiModel(nn.Module):
    def __init__(self, d_in, d_out):
        super(PiModel, self).__init__()
        self.l1 = nn.Linear(d_in, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, d_out)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.softmax(self.l3(x)/T, dim=-1)

        return x


def TrainModel(NNmodel, feature, label, op, gamma_t, delta, PG, criterion=torch.nn.MSELoss()):
    max_epochs = 1
    for epoch in range(1, max_epochs+1):
        # Training
        # Transfer to GPU
        local_batch = torch.tensor(feature).view(1, -1).to(device)
        local_labels = torch.tensor(label).view(1, -1).to(device)
        # Model computations
        out = NNmodel(local_batch)
        if PG:
            loss = PGLoss(out, local_labels, gamma_t, delta)
        else:
            loss = criterion(out, local_labels)

        op.zero_grad()
        loss.backward()
        op.step()

    return 


class VaModel(nn.Module):
    def __init__(self, d_in, d_out):
        super(VaModel, self).__init__()
        self.l1 = nn.Linear(d_in, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, d_out)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x

def PGLoss(output, target, gamma_t, delta):
    # print(target)
    loss = - torch.sum((torch.log(output)) * gamma_t * delta)

    return loss


class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here

        # Tips for TF users: You will need a function that collects the probability of action taken
        # actions; i.e. you need something like
        #
            # pi(.|s_t) = tf.constant([[.3,.6,.1], [.4,.4,.2]])
            # a_t = tf.constant([1, 2])
            # pi(a_t|s_t) =  [.6,.2]
        #
        # To implement this, you need a tf.gather_nd operation. You can use implement this by,
        #
            # tf.gather_nd(pi,tf.stack([tf.range(tf.shape(a_t)[0]),a_t],axis=1)),
        # assuming len(pi) == len(a_t) == batch_size

        self.s_dim = state_dims
        self.a_dim = num_actions
        self.model = PiModel(self.s_dim, self.a_dim)
        self.model.apply(init_weights)
        self.op = optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))

    def __call__(self,s) -> int:
        # TODO: implement this method
        self.model.eval()
        action = self.model(torch.tensor(s)).detach().numpy()
        # print(action)
        # input()
        return action

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t (int)
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # TODO: implement this method
        # a = F.one_hot(torch.tensor(a), num_classes=self.a_dim)
        a = torch.tensor(a, dtype=torch.float64)
        self.model.train()
        TrainModel(self.model, s, a, self.op, gamma_t, delta, PG=True)

        return


class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        # TODO: implement here
        self.s_dim = state_dims
        self.model = VaModel(self.s_dim, 1)
        self.model.apply(init_weights)
        self.op = optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))

    def __call__(self,s) -> float:
        # TODO: implement this method
        self.model.eval()
        output = self.model(torch.tensor(s)).detach().numpy()

        return float(output)

    def update(self,s,G):
        # TODO: implement this method
        self.model.train()
        delta = G
        gamma_t = 1
        TrainModel(self.model, s, delta, self.op, 0, 0, PG=False)

        return


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    # TODO: implement this method

    GList = []

    ### for each episode
    for episode in range(num_episodes):
        ### generate episode
        sList = []
        sList.append(env.reset())
        rList = [0]
        aList = []
        done = False
        while not done:
            a = pi(sList[-1])
            s_, r_, done, _ = env.step(a)
            rList.append(r_)
            sList.append(s_)
            aList.append(a)

        T = len(aList) ## 0 ~ T-1

        ### loop for each episode
        for t in range(T):
            G = 0
            for k in range(t+1, T+1):
                G += gamma**(k-t-1) * rList[k]
            if t == 0:
                GList.append(G) ## store G0
                print(episode, G)
            
            delta = G - V(sList[t])
            V.update(sList[t], delta)
            pi.update(sList[t], aList[t], gamma**t, delta)

    return GList

