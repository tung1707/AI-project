#create a single model and train it (use pytorch)
#dang training gi
#input la gi
#output la gi

import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(in_features=64,out_features=32),
    nn.Sigmoid(),
    nn.Linear(in_features=32,out_features=20),
    nn.Sigmoid(),
    nn.Linear(20,10),
    nn.Sigmoid(),
    
    nn.Linear(10,0),
    nn.Sigmoid()

)