from pyimagesearch import mlp
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import torch.nn as nn
import torch

def next_batch(inputs,targets,batchSize):
    for i in range(0,inputs.shape[0],batchSize):
        yield(inputs[i:i + batchSize],targets[i:i + batchSize])

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))

