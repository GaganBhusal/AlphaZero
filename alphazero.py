from collections import deque
import copy
import numpy as np
from tqdm import trange
import random

from MCTS import MCTS

import torch.nn as nn
import torch
from torch import optim

class AlphaZero:

    def __init__(self, env, model, policy_loss, value_loss, optimizer, device, PARAMS, current_player = 1):
        self.env = env
        self.model = model
        self.model.to(device)
        self.optimizer = optimizer
        self.mainBuffer = deque(maxlen = 500_000)
        self.policy_loss = policy_loss
        self.value_loss = value_loss
        self.device = device
        self.current_player = current_player
        self.tau = 1.25
        self.entropy_weight = 0.01
        self.PARAMS = PARAMS
        
        self.baseline_model = copy.deepcopy(model)
        self.baseline_model.to(device)
        self.baseline_model.eval()
        for p in self.baseline_model.parameters():
            p.requires_grad = False       
            
        self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    def selfPlayData(self, ):
        buffer = []
        state = self.env.reset()
        current_player = 1
        move_number = 0
        # tau = self.PARAMS['TAU']
        
        while True:
            if move_number < 10:
                tau = 1
            else:
                tau = 0.01
            root = MCTS(self.env, state, current_player, self.model, self.device, self.PARAMS)

            actions = root.search()
            temperature_probs = actions ** (1/tau)
            temperature_probs /= np.sum(temperature_probs)
            buffer.append((state, actions, current_player))
            action = np.random.choice(self.env.all_moves, p = temperature_probs)
            state, result, done = self.env.step(state, action, current_player)
            if done:
                mainBuffer = []
                for states, actions, player in buffer:
                    if result == 0:
                        value = 0
                    elif result == player:
                        value = 1
                    else:
                        value = -1
                    mainBuffer.append((self.env.stackedStates(states, player), actions, value))
                return mainBuffer

            current_player = -current_player
            move_number += 1

    def train(self):
        for i in trange(self.PARAMS['TOTAL_ITERATIONS']):
                
            self.model.eval()
            for _ in trange(self.PARAMS['SELF_PLAY_ITERATIONS']):
                self.mainBuffer.extend(self.selfPlayData())
            
            self.model.train()
            losses = []
            epoch_loss = 0
 
            for epoch in trange(self.PARAMS['EPOCHS']):

                epoch_loss = 0
                batch_count = 0
                np.random.shuffle(self.mainBuffer)
                for idx in range(0, len(self.mainBuffer), self.PARAMS["BATCH_SIZE"]):
                    # print(self.mainBuffer)
                    # print(idx)
                    data = random.sample(self.mainBuffer, self.PARAMS["BATCH_SIZE"])
                    states, actions, value = zip(*data)
    
                    actions = torch.tensor(np.array(actions), dtype = torch.float).to(self.device)
                    values = torch.tensor(np.array(value), dtype = torch.float).unsqueeze(1).to(self.device)
                    states = torch.tensor(np.array(states), dtype = torch.float).to(self.device)

                    model_policy, model_value = self.model(states)
                    
                    policy_probs = torch.softmax(model_policy, dim=1)
                    entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-10), dim=1).mean()
                    
                    policy_loss = self.policy_loss(model_policy, actions)
                    value_loss = self.value_loss(model_value, values)
                    
                    self.entropy_weight = 0.01
                    total_loss = policy_loss + value_loss - self.entropy_weight * entropy
                    
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()            

                    epoch_loss += total_loss.item()
                    batch_count += 1

                avg_loss = epoch_loss/batch_count
                    
                print(f"Epoch {epoch + 1}\tTotal Loss : {avg_loss}")
                losses.append(avg_loss)

                if avg_loss < 0.01:
                    print(f"Loss very low ({avg_loss:.6f}), stopping early to prevent overfitting")
                    break

                
            self.evaluation_low()
            # win_rate = self.evaluation()
            # if win_rate > 0.45:
            self.save_model(i)

            # if win_rate > 0.55:
            print("Baseline Model Changed!!!")
            self.baseline_model = copy.deepcopy(self.model)
            self.baseline_model.to(self.device)
            self.baseline_model.eval()
            for p in self.baseline_model.parameters():
                p.requires_grad = False
    
                

    def save_model(self, i):

        checkpoint = {
            "model" : self.model.state_dict(),
            "optimizer_state" : self.optimizer.state_dict()
        }
        torch.save(checkpoint, f'Checkpoints/checkpoint{i}.pth')
        print("Hit Checkpoint")
        
    def play_game(self, model1, model2):
        state = self.env.reset()
        current_player = 1

        while True:
            root = MCTS(self.env, state, current_player, model1, self.device, self.PARAMS, 1)
            actions = root.search()
            action = np.argmax(actions)
            state, result, done = self.env.step(state, action, current_player)
            if done:
                return result * current_player
            current_player = -current_player

            root = MCTS(self.env, state, current_player, model2, self.device, self.PARAMS, 1)
            actions = root.search()
            action = np.argmax(actions)
            state, result, done = self.env.step(state, action, current_player)
            if done:
                return result * -current_player
            current_player = -current_player
    

    def evaluation(self):
        wins = 0
        draws = 0
        self.model.eval()
        pmodel = self.baseline_model

        for game in trange(self.PARAMS['EVALUATION_GAMES']):
            if game%2 == 0:
                result = self.play_game(self.model, pmodel)
            else:
                result = -self.play_game(pmodel, self.model)

            if result == 1:
                wins += 1
            elif result == 0:
                draws += 1

        win_rate = wins/self.PARAMS["EVALUATION_GAMES"]
        print(f"Evaluation : {wins}/{self.PARAMS['EVALUATION_GAMES']} Rate : {win_rate} Draws : {draws}")
        # return win_rate

    def random_agent(self, model1):
        state = self.env.reset()
        current_player = 1

        while True:
            root = MCTS(self.env, state, current_player, model1, self.device, self.PARAMS, 1)
            actions = root.search()
            action = np.argmax(actions)
            state, result, done = self.env.step(state, action, current_player)
            if done:
                return result * current_player
            current_player = -current_player

            # root = MCTS(self.env, state, current_player, model2, self.device, self.PARAMS, 1)
            # actions = root.search()
            # action = np.argmax(actions)
            actions = np.random.choice(self.env.get_valid_moves(state))
            state, result, done = self.env.step(state, action, current_player)
            if done:
                return result * -current_player
            current_player = -current_player

    def evaluation_low(self):
        wins = 0
        draws = 0
        self.model.eval()
        pmodel = self.baseline_model

        for game in trange(self.PARAMS['EVALUATION_GAMES']):
            if game%2 == 0:
                result = self.play_game(self.model, pmodel)
            else:
                result = self.random_agent(self.model)

            if result == 1:
                wins += 1
            elif result == 0:
                draws += 1


        win_rate = wins/self.PARAMS["EVALUATION_GAMES"]
        print(f"Evaluation with random: {wins}/{self.PARAMS['EVALUATION_GAMES']} Rate : {win_rate} Draws : {draws}")
        # return win_rate
                            