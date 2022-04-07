from tkinter.messagebox import NO
from gym import Env
import numpy as np

expParaList = [0.4, 1.9, 4.4, 2.5, 3.4, 0.7]

# np.random.seed(42)

class Site:
    def __init__(self, number):
        self.number = number
        self.c = float(np.random.randint(low=20, high=50))
        self.dataPatternd()

    def dataStreamD(self):
        D = np.random.uniform(low=0.01*self.c, high=0.15*self.c)
        self.D = D
        return D

    def dataPatternd(self):
        ### exponential distribution
        # expPara = 1/np.random.uniform(low=0.01*self.c, high=0.10*self.c) # <=1
        self.expPara = expParaList[self.number]
        return self.expPara
    
    def dataStreamd(self):
        d = np.random.exponential(1/self.expPara)
        # print("lambda and d: ", self.expPara, d)
        return d

class SiteEnv(Env):
    def __init__(self, numSites, renderTF=False):
        ### generate sites
        self.numSites = numSites
        cList = []
        self.siteList = []
        for i in range(numSites):
            site = Site(i)
            self.siteList.append(site)
            cList.append(site.c)
        self.cList = cList

        ### continuous state space, shape: (numSites+1,)
        ### last component is the sum of current D: \sum_i site[i].D

        self.observation_space = np.zeros(numSites + 1)
        # self.observation_space = Box(low=np.zeros(numSites + 1), high=np.array(cList+[7*numSites]), shape=(numSites+1,))
        
        ### continuous action space, shape: (numSites,)
        ### constraint: action should sum  to \sum_i site[i].D
        self.action_space = np.zeros(numSites)
        
        ### current state 
        self.state = self.reset()
        
        ### number of rounds
        self.rounds = 10
        self.current_round = 0
        
        ### reward collected
        self.cumulated_reward = 0

        ### rener print state info or not
        self.renderTF = renderTF
    
    def step(self, action):
        done = False
        info = {"Hello world."}
        self.current_round += 1

        next_state = np.zeros_like(self.state)
        
        ### transition
        sumd = self.calculated()
        sumD, _ = self.calculateD()
        next_state[:-1] = self.state[:-1] + action + sumd

        _ = np.clip(next_state[:-1] - self.cList, 0, a_max=None)
        reward = 1 - np.sum(_)
        self.cumulated_reward += reward

        self.state[-1] = sumD
        self.state[:-1] = np.clip(next_state[:-1], 0, a_max=self.cList)

        if self.renderTF:
            self.render(action, reward) 
        done = True if self.current_round == self.rounds else False
            
        return self.state, self.cumulated_reward, done, info
    
    def reset(self):
        self.state = np.zeros(self.numSites + 1)
        sumD, _ =self.calculateD()
        self.state[-1] = sumD
        return self.state

    def render(self, action, reward):
        print(f"Round : {self.current_round}\nAction: {action}\nState: {self.state}\nCapacity: {self.cList}")
        print(f"\nReward Received: {reward}\nTotal Reward : {self.cumulated_reward}")
        print("=============================================================================")

    def calculateD(self):
        DList = []
        for site in self.siteList:
            D = site.dataStreamD()
            DList.append(D)
        return np.sum(DList), DList

    def calculated(self):
        dList = []
        for site in self.siteList:
            d = site.dataStreamd()
            dList.append(d)
        return np.array(dList)


if __name__ == "__main__":
    env = SiteEnv(3, True)
    done = False
    state = env.reset()
    while not done:
        action = np.random.uniform(low=0, high=100, size=env.action_space.shape) ### random action 
        action = state[-1] * action/np.sum(action) ### normalize

        state, reward, done, info = env.step(action)