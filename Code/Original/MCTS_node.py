import numpy as np

class Node:

    def __init__(self, state, parent, move, player, prob = 0):
        self.H = 6
        self.W = 7
        self.state = state
        self.parent = parent
        self.move = move

        self.player = player
        self.children = {}
        
        self.prob = prob
        
        self.N = 0
        self.W = 0
        
    def is_fully_expanded(self):
        return len(np.argwhere(self.state[0] == 0)) == 0
# returns True if all nodes are expanded else False
    
    def UCB1(self, c = 2):
        if self.N == 0:
            return float('inf')

        # Not sure about this change
        # (Have to look later too)
        q_value = -self.W / self.N
        return q_value + c * self.prob * np.sqrt(self.parent.N)/(self.N + 1)