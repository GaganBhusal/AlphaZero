import torch
from MCTS_node import Node
import numpy as np

class MCTS:

    def __init__(self, env, root_state, root_player, model, device, PARAMS, which = None):
        self.env = env
        self.H = 6
        self.W = 7
        self.policy = 0
        self.model = model
        self.device = device
        self.root = Node(state = root_state,
                         parent = None,
                         move = None,
                         player = root_player
                        )
        self.PARAMS = PARAMS
        self.which = which
        
    def selection(self):
        current = self.root
        while True:
            is_terminal, _ = self.env.is_terminal(current.state)
            if is_terminal:
                return current
                
            if len(current.children) == 0:
                return current
                
            # if not current.is_fully_expanded():
            #     print(f"Returend From Selection : {current}")
            #     return current
                
            best_child = max(current.children.values(), key = lambda c: c.UCB1())
            current = best_child
            
            
        # return current

    def expansion(self, node, policy):
        valid_states = self.env.get_valid_moves(node.state)

        for action, prob in enumerate(policy):
            if valid_states[action] == 1 and action not in node.children:
                new_state = self.env.step(node.state, action, node.player)[0]
                child = Node(state = new_state,
                             parent = node,
                             move = action,
                             player = -node.player,
                             prob = prob
                            )
                node.children[action] = child
                
    def backpropagation(self, child, value):
        current = child

        while current is not None:
            current.N += 1
            current.W += value
            value = -value
            current = current.parent

    @torch.no_grad()
    def search(self):

        searches = self.PARAMS["SEARCHES"]
        if self.which is not None:
            searches = self.PARAMS["EVALUATION_SEARCHES"]
        
        
        for _ in range(searches):
            node = self.selection()
            is_terminal, value = self.env.is_terminal(node.state)
            
            if is_terminal:
                value = value * node.player
            else:
                stackedStates = self.env.stackedStates(node.state, node.player)
                policy, value = self.model(torch.tensor(stackedStates, dtype = torch.float).unsqueeze(0).to(self.device))
                policy = torch.softmax(policy, dim = 1).squeeze(0).cpu().detach().numpy()
                if node is self.root:
                    alpha = self.PARAMS['ALPHA']
                    eps = self.PARAMS['EPSILON']
                    noise = np.random.dirichlet([alpha] * len(self.env.all_moves))
                    policy = (1 - eps) * policy + eps * noise

                value = value.squeeze(0).cpu().item()
                valid_moves = self.env.get_valid_moves(node.state)
                policy = policy * valid_moves
                if np.sum(policy)>0:
                    policy = policy/(np.sum(policy))
                else:
                    policy = valid_moves/np.sum(valid_moves)


                self.expansion(node, policy)
                # if len(node.children) > 0:
                #     best_action = max(node.children.keys(), key=lambda a: node.children[a].prob)
                #     child = node.children[best_action]
                # else:
                #     child = node
            target_node = node
    
            self.backpropagation(target_node, value)
        actions = np.zeros(7)

        if len(self.root.children) == 0:
            valid_moves = self.env.get_valid_moves(self.root.state)
            action_probs = np.array(valid_moves / np.sum(valid_moves))
            return action_probs

        total_visits = sum(child.N for child in self.root.children.values())
        if total_visits > 0:
            for action, child in self.root.children.items():
                actions[action] = child.N
            actions /= total_visits
        else:
            valid_moves = self.env.get_valid_moves(self.root.state)
            actions = np.array(valid_moves / np.sum(valid_moves))

        return actions

    def __repr__(self):
        return f"{self.root.state}\n{self.root.parent}\n{self.root.children}\n{self.root.action}\n{self.root.children.n}"