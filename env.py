from multiprocessing.dummy import active_children
from pydoc import render_doc
from gym import Env
import numpy as np

expParaList = [0.9, 1.9, 2.4, 2.5, 3.4, 4.7]
cList = [25., 34., 27., 29., 32., 43.]

actionList = np.array([
    [1,0,0], [0,1,0], [0,0,1], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5], [1/3, 1/3, 1/3]
])

class Site:
    def __init__(self, number):
        self.number = number
        self.c = cList[self.number]
        self.dataPatternd()

    def dataStreamD(self):
        D = np.random.uniform(low=0.01*self.c, high=0.05*self.c)
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
    def __init__(self, numSites, renderTF=False, numActList=5):
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
        self.generateActionList()
        self.action_space = np.zeros(self.actionList.shape[0])

        
        ### current state 
        self.state = self.reset()

        ### render: print state info or not
        self.renderTF = renderTF

    def generateActionList(self, randomSeed=42):
        # np.random.seed(randomSeed)
        # act = np.random.randint(low=0, high=4, size=(self.action_space.shape[0], self.numSites))
        # self.actionList = act/act.sum(axis=1)[:, np.newaxis]
        self.actionList = actionList
        # print("Action Space: ", self.actionList.shape)
        # print("Action List:\n", self.actionList)
        # input()

    def rewardStructure(self, action):
        _ = self.state[:-1] + action + self.sumd
        reward = 0
        noExceed = True
        for i in range(self.numSites):
            if _[i] > self.cList[i]: # exceed buffer
                reward -= 1
                noExceed = False
        if noExceed:
            reward += 0
            if self.current_round == self.rounds:
                reward += (self.numSites)
        # _ = np.clip(_ - self.cList, a_min=0, a_max=None)
        # reward = 2 - np.sum(_)
        return reward

    def step(self, actionNum):
        done = False
        info = {"Hello world."}
        self.current_round += 1
        if isinstance(actionNum, int):
            action = self.actionList[actionNum] ### choose action
        else:
            action = actionNum
        action = self.state[-1] * action / np.sum(action) ### normalize

        next_state = np.zeros_like(self.state)
        
        ### transition
        sumD, _ = self.calculateD()
        next_state[:-1] = self.state[:-1] + action + self.sumd
        next_state[-1] = sumD
        next_state[:-1] = np.clip(next_state[:-1], a_min=0, a_max=self.cList)

        ### Reward
        reward = self.rewardStructure(action)
        self.cumulated_reward += reward
        
        if self.renderTF: # and self.current_round == self.rounds:
            self.render(action, actionNum, reward)

        ### Done or not
        done = True if self.current_round == self.rounds  else False
        self.sumd = self.calculated()
        # if done:
        #     self.render(action, actionNum, reward)
        
        self.state = next_state

        return next_state, reward, done, info
    
    def reset(self):
        ### initial state
        newState = np.zeros(self.numSites + 1)
        sumD, _ = self.calculateD()
        self.sumd = self.calculated()
        newState[-1] = sumD
        self.state = newState

        ### number of rounds
        self.rounds = 15
        self.current_round = 0
        
        ### reward collected
        self.cumulated_reward = 0
        return newState

    def render(self, action, actionNum, reward):
        print(f"Round : {self.current_round}\nAction: {action, actionNum}\nState: {self.state}\nCapacity: {self.cList}")
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