from gym import Env
import numpy as np

expParaList = [0.4, 1.9, 4.4, 2.5, 3.4, 0.7]
cList = [25., 34., 50., 29., 32., 43.]

# np.random.seed(42)


class Site:
    def __init__(self, number):
        self.number = number
        self.c = cList[self.number]
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
    def __init__(self, numSites, renderTF=False, numList=20):
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
        self.action_space = np.zeros(numList)
        self.generateActionList()
        
        ### current state 
        self.state = self.reset()

        ### rener: print state info or not
        self.renderTF = renderTF

    def generateActionList(self, randomSeed=42):
        np.random.seed(randomSeed)
        act = np.random.rand(self.action_space.shape[0], self.numSites)
        self.actionList = act/act.sum(axis=1)[:, np.newaxis]
        # print("Action space: ", self.actionList.shape)
        # print("Action List: ", self.actionList)
    
    def step(self, action):
        done = False
        info = {"Hello world."}
        self.current_round += 1
        action = self.actionList[action] ### choose action
        print("Action: ", action)
        action = self.state[-1] * action / np.sum(action) ### normalize

        next_state = np.zeros_like(self.state)
        
        ### transition
        sumd = self.calculated()
        sumD, _ = self.calculateD()
        next_state[:-1] = self.state[:-1] + action + sumd

        _ = np.clip(next_state[:-1] - self.cList, a_min=0, a_max=None)
        reward = 1 - np.sum(_)
        self.cumulated_reward += reward
        
        if self.renderTF:
            self.render(action, reward) 

        self.state[-1] = sumD
        self.state[:-1] = np.clip(next_state[:-1], a_min=0, a_max=self.cList)

        done = True if self.current_round == self.rounds else False
            
        return self.state, self.cumulated_reward, done, info
    
    def reset(self):
        ### initial state
        self.state = np.zeros(self.numSites + 1)
        sumD, _ = self.calculateD()
        self.state[-1] = sumD

        ### number of rounds
        self.rounds = 10
        self.current_round = 0
        
        ### reward collected
        self.cumulated_reward = 0
        return self.state

    def render(self, action, reward):
        print(f"Round : {self.current_round}\nAction: {action}\nState: {self.state}\nCapacity: {self.cList}")
        print(f"\nReward Received: {reward}\nTotal Reward : {self.cumulated_reward}")
        print("=============================================================================")
        if np.any(np.isnan(action)):
            print(np.isnan(action), np.any(np.isnan(action)))
            input()


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
    for i in range(10):
        done = False
        state = env.reset()

        while not done:
            action = np.random.uniform(low=0, high=100, size=env.action_space.shape) ### random action 

            state, reward, done, info = env.step(action)