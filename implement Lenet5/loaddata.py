import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

batch_size=64
num_classes = 10
learning_rate=0.001
num_epochs=10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = torchvision.datasets.MNIST(root = './data',
                                           train=True,
                                           transform=transforms.Compose([
                                               transforms.Resize((32,32)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.1307,),std=(0.3081,))]),download=True)


test_dataset = torchvision.datasets.MNIST(root = './data',
                                           train=False,
                                           transform=transforms.Compose([
                                               transforms.Resize((32,32)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.1325,),std=(0.3105,))]),download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

