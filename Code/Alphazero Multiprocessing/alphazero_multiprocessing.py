from collections import deque
import copy
import numpy as np
from tqdm import tqdm
import random
import time
from MCTS import MCTS
import torch.multiprocessing as mp
from functools import partial
import torch.nn as nn
import torch
from torch import optim
import os
from torch.utils.tensorboard import SummaryWriter



os.makedirs("Checkpoints", exist_ok=True)

def self_play_worker(local_env, model, device, params, game_id):

    buffer = []
    state = local_env.reset()
    current_player = 1
    move_number = 0
    
    while True:
        if move_number < 10:
            tau = 1
        else:
            tau = 0.01

        root = MCTS(local_env, state, current_player, model, device, params)
        actions = root.search()
        
        temperature_probs = actions ** (1/tau)
        temperature_probs /= np.sum(temperature_probs)
        
        buffer.append((state, actions, current_player))
        
        # ADDED HORIZONTAL FLIP TO DOUBLE THE TRAINING DATA
        flipped_state = np.flip(state, axis=1)
        flipped_actions = np.flip(actions)
        buffer.append((flipped_state, flipped_actions, current_player))

        action = np.random.choice(local_env.all_moves, p=temperature_probs)
        state, result, done = local_env.step(state, action, current_player)
        
        if done:
            main_buffer = []
            for states, acts, player in buffer:
                if result == 0:
                    value = -0.1 #Penalize a little for draw
                elif result == player:
                    value = 1
                else:
                    value = -1
                main_buffer.append((local_env.stackedStates(states, player), acts, value))
            return main_buffer

        current_player = -current_player
        move_number += 1

def evaluation_worker(local_env, model, baseline_model, device, params, worker_id):

    state = local_env.reset()
    current_player = 1
    
    if worker_id % 2 == 0:
        model1, model2 = model, baseline_model
    else:
        model1, model2 = baseline_model, model

    while True:
        root = MCTS(local_env, state, current_player, model1, device, params, 1)
        actions = root.search()
        action = np.argmax(actions)
        state, result, done = local_env.step(state, action, current_player)
        if done:
            result *= current_player
            if worker_id % 2 == 0:
                return result
            return -result
        
        current_player = -current_player

        root = MCTS(local_env, state, current_player, model2, device, params, 1)
        actions = root.search()
        action = np.argmax(actions)
        state, result, done = local_env.step(state, action, current_player)
        if done:
            result *= -current_player
            if worker_id % 2 == 0:
                return result
            return -result
        
        current_player = -current_player


class AlphaZero:

    def __init__(self, env, model, policy_loss, value_loss, optimizer, device, PARAMS, current_player = 1):
        self.env = env
        self.model = model
        self.model.to(device)
        self.optimizer = optimizer
        self.mainBuffer = deque(maxlen = 20_000)
        self.policy_loss = policy_loss
        self.value_loss = value_loss
        self.device = device
        self.current_player = current_player
        self.tau = 1.25
        self.entropy_weight = 0.01
        self.PARAMS = PARAMS
        self.entropy_weight = 0.01

        self.avg_loss_per_epoch = []
        self.policy_loss_per_epoch = []
        self.value_loss_per_epoch = []
        self.entropy_per_epoch = []
        self.win_rate = []

        self.writer = SummaryWriter('experiments/Experiment1')
        self.baseline_model = copy.deepcopy(model)
        self.baseline_model.to(device)
        self.baseline_model.eval()
        for p in self.baseline_model.parameters():
            p.requires_grad = False       
            
        self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    """THis function is moved outside the class!!"""
    # def selfPlayData(self, ):
    #     buffer = []
    #     state = self.env.reset()
    #     current_player = 1
    #     move_number = 0
    #     # tau = self.PARAMS['TAU']
    #     while True:
    #         if move_number < 10:
    #             tau = 1
    #         else:
    #             tau = 0.01

    #         root = MCTS(self.env, state, current_player, self.model, self.device, self.PARAMS)

    #         actions = root.search()
    #         temperature_probs = actions ** (1/tau)
    #         temperature_probs /= np.sum(temperature_probs)
    #         buffer.append((state, actions, current_player))

    #         # ADDED HORIZONTAL FLIP TO DOUBLE THE TRAINING DATA
    #         flipped_state = np.flip(state, axis=1) 
    #         flipped_actions = np.flip(actions) 
    #         buffer.append((flipped_state, flipped_actions, current_player))

    #         action = np.random.choice(self.env.all_moves, p = temperature_probs)
    #         state, result, done = self.env.step(state, action, current_player)
    #         if done:
    #             # print(f"Game ID {gam/e_id}")
    #             mainBuffer = []
    #             for states, actions, player in buffer:
    #                 if result == 0:
    #                     value = 0
    #                 elif result == player:
    #                     value = 1
    #                 else:
    #                     value = -1
    #                 mainBuffer.append((self.env.stackedStates(states, player), actions, value))
    #             return mainBuffer

    #         current_player = -current_player
    #         move_number += 1

    def train(self):
        with tqdm(total = self.PARAMS["TOTAL_ITERATIONS"]) as pbar1:
            for i in range(self.PARAMS['TOTAL_ITERATIONS']):
                # print("==" * 45)
                # print(f"Training Iteration {i}")
                
                # self.model.to("cpu")
                self.model.eval()

                self.model.share_memory()

                total_workers = 12
                

                with mp.Pool(processes=total_workers) as pool:
                    worker_func = partial(self_play_worker, self.env, self.model, self.device, self.PARAMS)
                    buffer_data = pool.map(worker_func, range(self.PARAMS['SELF_PLAY_ITERATIONS']))
                
                
                
                for data in buffer_data:
                    self.mainBuffer.extend(data)


                self.model.to(self.device)
                self.model.train()
                losses = []

                sum_epoch_loss = 0
                sum_policy_loss = 0
                sum_value_loss = 0
                sum_entropy = 0


                # with tqdm(total = self.PARAMS["EPOCHS"]) as pbar2:
                for epoch in range(self.PARAMS['EPOCHS']):

                    epoch_loss = 0
                    batch_count = 0
                    np.random.shuffle(self.mainBuffer)
                    for idx in range(0, len(self.mainBuffer), self.PARAMS["BATCH_SIZE"]):

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
                        
                        
                        total_loss = policy_loss + value_loss - self.entropy_weight * entropy
                        sum_epoch_loss += total_loss.item()
                        sum_policy_loss += policy_loss.item()
                        sum_value_loss += value_loss.item()
                        sum_entropy += entropy.item()

                        self.optimizer.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()            

                        epoch_loss += total_loss.item()
                        batch_count += 1

                    avg_loss = epoch_loss/batch_count
                    avg_policy_loss = sum_policy_loss/batch_count
                    avg_value_loss = sum_value_loss/batch_count
                    avg_entropy_loss = sum_entropy/batch_count
                        
                    # print(f"\tEpoch {epoch}\tTotal Loss : {avg_loss}")
                    losses.append(avg_loss)

                    self.avg_loss_per_epoch.append(avg_loss)
                    self.policy_loss_per_epoch.append(avg_policy_loss)
                    self.value_loss_per_epoch.append(avg_value_loss)
                    self.entropy_per_epoch.append(avg_entropy_loss)

                    self.writer.add_scalars(
                        main_tag="Losssss",
                        tag_scalar_dict={
                            "Average Loss" : avg_loss,
                            "Policy Loss" : avg_policy_loss, 
                            "Value Loss" : avg_value_loss,
                            "Entropy Loss" : avg_entropy_loss,
                            },     
                        global_step=i*epoch+1           
                    )

                        # pbar2.set_postfix(AverageLoss = avg_loss, PolicyLoss = avg_policy_loss, ValueLoss = avg_value_loss, EntropyLoss = avg_entropy_loss)
                        # pbar2.update(1)
                    
                # self.evaluation_low()
                win_rate, draws = self.evaluation()
                # if win_rate > 0.45:
                self.save_model(i)

                # if win_rate > 0.55:
                # print("Baseline Model Changed!!!")
                self.baseline_model = copy.deepcopy(self.model)
                self.baseline_model.to(self.device)
                self.baseline_model.eval()
                for p in self.baseline_model.parameters():
                    p.requires_grad = False

                progress_bar_data = {
                    "Win Rate" : win_rate,
                    "Draws" : draws,
                    "Average Loss" : avg_loss,
                    "Policy Loss" : avg_policy_loss, 
                    "Value Loss" : avg_value_loss,
                    "Entropy Loss" : avg_entropy_loss,
                    }                

                self.writer.add_scalars(
                    main_tag="Training Data", 
                    tag_scalar_dict={
                        "Win Rate" : win_rate,
                        "Draws" : draws,
                    },
                    global_step=i
                )                

                pbar1.set_postfix(progress_bar_data, refresh=True)
                pbar1.update(1)

    def save_model(self, i):

        checkpoint = {
            "model" : self.model.state_dict(),
            "optimizer_state" : self.optimizer.state_dict()
        }
        torch.save(checkpoint, f'Checkpoints/checkpoint{i}.pth')
        # print(f"\nSaved Checkpoint {i}!!\n")
        # print("==" * 90)
        # print("\n\n\n")
        

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

        self.model.eval()
        self.baseline_model.eval()
        self.model.share_memory()
        self.baseline_model.share_memory()

        total_workers = 12
        
        with mp.Pool(processes=total_workers) as pool:
            eval_func = partial(evaluation_worker, self.env, self.model, self.baseline_model, self.device, self.PARAMS)
            results = pool.map(eval_func, range(self.PARAMS['EVALUATION_GAMES']))

        wins = results.count(1)
        draws = results.count(0)
        losses = results.count(-1)

        win_rate = wins / self.PARAMS["EVALUATION_GAMES"]
        return win_rate, draws
