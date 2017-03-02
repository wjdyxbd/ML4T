"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_states=num_states
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.alpha=alpha
        self.gamma=gamma
        self.rar=rar
        self.radr=radr
        self.dyna=dyna
        self.state_q=np.zeros((self.num_states,self.num_actions))
        self.T=np.zeros((self.num_states,self.num_actions))
        self.R=np.zeros((self.num_states,self.num_actions))
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = rand.randint(0, self.num_actions-1)
        self.a=action
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        rar=self.rar
        radr=self.radr
        prand=rand.random()
        if prand < rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            array=list(self.state_q[s_prime])
            index=[]
            for idx,item in enumerate(array):
                if item >= max(array)-0.00001:
                    index.append(idx)
            index=np.array(index)
            if len(index)>0:
                action = np.random.choice(index)
        rar=rar*radr
        self.rar=rar
        self.state_q[self.s,self.a]=(1-self.alpha)*self.state_q[self.s,self.a]+self.alpha*(r+self.gamma*np.max(self.state_q[s_prime]))
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r

        self.R[self.s,self.a]=(1-self.alpha)*self.R[self.s,self.a]+self.alpha*r
        self.T[self.s,self.a]=s_prime
        self.s=s_prime
        self.a=action
        for i in range(self.dyna):
            self.hallucination()
        return action
    def hallucination(self):
        s=rand.randint(0,self.num_states-1)
        action=rand.randint(0,self.num_actions-1)
        s_prime=self.T[s,action]
        if (np.sum(self.T[s,action]))!=0 and s_prime>0:
            r=self.R[s,action]
            self.state_q[s,action]=(1-self.alpha)*self.state_q[s,action]+self.alpha*(r+self.gamma*np.max(self.state_q[s_prime]))

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
