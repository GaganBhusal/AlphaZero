from Resnet_model import Resnet
from connect4 import Connect4
from torch import optim
import numpy as np
import torch
from alphazero import AlphaZero
import torch.nn as nn

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
    "EVALUATION_SEARCHES" : 600,
    
    "ALPHA" : 1.4,
    "EPSILON" : 0.25,
    "TAU" : 1
    }


env = Connect4()

model = Resnet(PARAMETERS['IN_CHANNELS'], 
               PARAMETERS['OUT_CHANNELS'], 
               PARAMETERS['RESNET_BLOCKS']
              )

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