from Resnet_model import Resnet
from connect4 import Connect4
from torch import optim
import numpy as np
import torch
from alphazero import AlphaZero
import torch.nn as nn
from MCTS import MCTS

PARAMETERS = {
    
    "IN_CHANNELS" : 3,
    "OUT_CHANNELS" : 64,
    "RESNET_BLOCKS" : 5,
    
    "SELF_PLAY_ITERATIONS" : 50,
    "EPOCHS" : 10,
    "BATCH_SIZE" : 128,
    "TOTAL_ITERATIONS" : 200,
    
    "SEARCHES" : 800,
    "EVALUATION_GAMES" : 50,
    "EVALUATION_SEARCHES" : 1000,
    
    "ALPHA" : 0.6,
    "EPSILON" : 0.25,
    "TAU" : 1
    }


env = Connect4()


cp1 = 1
cp2 = 6

model1 = Resnet(PARAMETERS['IN_CHANNELS'], 
               PARAMETERS['OUT_CHANNELS'], 
               PARAMETERS['RESNET_BLOCKS']
              )


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model_checkpoint1 = torch.load(f"Checkpoints/checkpoint{cp1}.pth", map_location= "cpu")
model1.load_state_dict(model_checkpoint1["model"])
model1.to(device)

model1.eval()


model2 = Resnet(PARAMETERS['IN_CHANNELS'], 
               PARAMETERS['OUT_CHANNELS'], 
               PARAMETERS['RESNET_BLOCKS']
              )


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model_checkpoint = torch.load(f"Checkpoints/checkpoint{cp2}.pth", map_location= "cpu")
model2.load_state_dict(model_checkpoint["model"])
model2.to(device)

model2.eval()


state = env.reset()
current_player = 1

while True:
    root = MCTS(env, state, current_player, model1, device, PARAMETERS)
    actions = root.search()
    action = np.argmax(actions)
    state, result, done = env.step(state, action, current_player)

    print("==" * 70)
    print(f"\n\n{cp1} Move")
    env.show(state)

    if done:
        if result == current_player:
            # wins += 1
            print(f"Checkpoint {cp1} Won!!")
        break
    current_player = -current_player

    root = MCTS(env, state, current_player, model2, device, PARAMETERS)
    actions = root.search()
    action = np.argmax(actions)
    state, result, done = env.step(state, action, current_player)

    print("\n\n")
    print(f"{cp2} Move\n\n")
    env.show(state)

    if done:
        if result == current_player:
            # wins += 1
            print(f"Checkpoint {cp2} Won!!")
        break
    current_player = -current_player
    # print(f"Available Moves : {np.where(env.get_valid_moves(state)==1)}")
    # action = int(input("Enter action (0-6) : "))
    # state, result, done = env.step(state, action, current_player)
    # if done:
    #     if result == current_player:
    #         print("You Won!!")
    #     break
    # current_player = -current_player
    # env.show(state)