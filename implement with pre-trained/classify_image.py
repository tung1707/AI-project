from pyimagesearch import config
from torchvision import models
import numpy as np
import argparse
import torch
import cv2

def preprocess_image(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(config.IMAGE_SIZE,config.IMAGE_SIZE))
    image = image.astype("float32") / 255.0

    image -= config.MEAN
    image /= config.STD
    image = np.transpose(image,(2,0,1))
    image = np.expand_dims(image,0)
    return image

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to the input image")
ap.add_argument("-m","--model",type=str,default="vgg16",choices=["vgg16","vgg19","inception","densenet","resnet"],
                help = "name of pre-trained network to use")
args = vars(ap.parse_args())

MODELS = {
    "vgg16":models.vgg16(pretrained=True),
    "vgg19":models.vgg19(pretrained=True),
    "inception":models.inceptin_v3(pretrained=True),
    "densenet":models.densenet121(pretrained=True),
    "resnet":models.resnet50(pretrained=True)
}

print("[INFO] loading {} ...".format(args["model"]))
model = MODELS[args["model"]].to(config.DEVICE)
model.eval()

print("[INFO] loading image...")
image = cv2.imread(args["image"])
orig = image.copy()
image = preprocess_image(image)

image = torch.from_numpy(image)
image = image.to(config.DEVICE)

print("[INFO] loading ImageNet labels...")
imagenetLabels = dict(enumerate(open(config.IN_LABELS)))
#prediction
print("[INFO] classifying image with '{}'...".format(args["model"]))
logits = model(image)
probabilities = torch.nn.Softmax(dim=-1)(logits)
sortedProba = torch.argsort(probabilities,dim=-1,descending=True)

for(i,idx) in enumerate(sortedProba[0,:5]):
    print("{}. {}: {:.2f}%".format(i,imagenetLabels[idx.item()].strip(),probabilities[0,idx.item()]*100))

(label,prob)= (imagenetLabels[probabilities.argmax().item()],probabilities.max().item())
cv2.putText(orig,"Label: {},{:.2f}%".format(label.strip(),prob*100),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
cv2.imshow("Classification",orig)
cv2.waitKey(0)





