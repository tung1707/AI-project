import torch

def preprocess_input():
    image = torch.randn(3,418,278)
    vector = image.view(-1)
    vector = torch.flatten(image)
    print(vector)

preprocess_input()

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

resnet = models.resnet18(pretrained=True)
resnet.eval()

image = Image.open("university.jpg")
transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

image_tensor = transforms(image).unsqeeze(0)



