from Resnet_model import Resnet
from connect4 import Connect4
from torch import optim
import numpy as np
import torch
from alphazero_multiprocessing import AlphaZero
import torch.nn as nn
import torch.multiprocessing as mp


PARAMETERS = {
    
    "IN_CHANNELS" : 3,
    "OUT_CHANNELS" : 128,
    "RESNET_BLOCKS" : 5,
    
    "SELF_PLAY_ITERATIONS" : 5, #100
    "EPOCHS" : 5,
    "BATCH_SIZE" : 128,
    "TOTAL_ITERATIONS" : 50,  #200
    
    "SEARCHES" : 800,
    "EVALUATION_GAMES" : 20,
    "EVALUATION_SEARCHES" : 600,
    
    "ALPHA" : 1.4,
    "EPSILON" : 0.25,
    "TAU" : 1
    }





if __name__ == "__main__":
    mp.set_start_method("spawn", force = True)

    env = Connect4()

    model = Resnet(PARAMETERS['IN_CHANNELS'], 
                PARAMETERS['OUT_CHANNELS'], 
                PARAMETERS['RESNET_BLOCKS']
                )
    
    model.share_memory()

    optimizer = optim.Adam(model.parameters(), lr = 0.0008, weight_decay = 1e-4)
    policy_loss = nn.CrossEntropyLoss()
    value_loss = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent= AlphaZero(env, 
                    model, 
                    policy_loss, 
                    value_loss, 
                    optimizer, 
                    device, 
                    PARAMETERS
                    )

    agent.train()