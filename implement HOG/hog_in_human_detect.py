from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


hog = cv2.HOGDescriptor()#khoi tao 1 bo mo ta dac trung theo thuat toan HOG
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())#thiet lap mo hinh pre-trained dua vao thuat toan SVM

import glob
import matplotlib.patches as patches
import cv2
import imutils 
import matplotlib.pyplot as plt

for i ,imagePath in enumerate(glob.glob('images/*.bmp')):
  if i <= 6:
    image = cv2.imread(imagePath)
    image = imutils.resize(image,width=min(400,image.shape[1]))
    orig = image.copy()

    plt.figure(figsize=(8,6))
    ax1 = plt.subplot(1,2,1)
    (rects,weights) = hog.detectMultiScale(img=image,winStride=(4,4),padding=(8,8),scale = 1.5)
    print('weights: ',weights)

    for(x,y,h,w) in rects:
      rectFig = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
      ax1.imshow(orig)
      ax1.add_patch(rectFig)
      plt.title("Anh truoc non max suspression")
    
    rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
    print('rects: ',rects.shape)

    pick = non_max_suppression(rects,probs=None,overlapThresh=0.65)
    ax2 = plt.subplot(1,2,2)
    for(xA,yA,xB,yB) in pick:
      for(xA,yA,xB,yB) in pick:
        w = xB-xA
        h = yB-yA

        plt.imshow(image)
        plt.title("Anh sau non max suppression")
        rectFig = patches.Rectangle((xA,yA),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax2.add_patch(rectFig)
      
      filename = imagePath[imagePath.rfind("\\") + 1:]
      print("[INFO] {}:{} orginal boxes,{}after suspression".format(filename,len(rects),len(pick)))