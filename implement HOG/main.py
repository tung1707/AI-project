# bieu dien HOG su dung cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = plt.imread('pic.jpg',cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print('image shape:',img.shape)
print('gray shape: ',gray.shape)

plt.figure(figsize=(16,4))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title('Orginal Image')
plt.subplot(1,2,2)
plt.imshow(gray)
plt.title('Gray Image')


gx = cv2.Sobel(gray,cv2.CV_32F,dx=0,dy=1,ksize=3)#cv2.32f=ddepth=depth of output imagez ,ksize=kernel size
gy = cv2.Sobel(gray,cv2.CV_32F,dx=1,dy=0,ksize=3)

print('gray shape: {}'.format(gray.shape))
print('gx shape: {}'.format(gx.shape))
print('gy shape: {}'.format(gy.shape))
g,theta = cv2.cartToPolar(gx,gy,angleInDegrees=True)
print('gradient format:{}'.format(g.shape))
print('theta format:{}'.format(theta.shape))

w = 20
h = 10
plt.figure(figsize=(w,h))
plt.subplot(1,4,1)
plt.title("gradient of x")
plt.imshow(gx)

plt.subplot(1,4,2)
plt.title("gradient of y")
plt.imshow(gy)

plt.subplot(1,4,3)
plt.title("Magnitude of gradient")
plt.imshow(g)

plt.subplot(1,4,4)
plt.title("Direction of gradient")
plt.imshow(theta)

print("Original image shape ",img.shape)
cell_size = (8,8)
block_size = (2,2)
nbins = 9


winSize = (img.shape[1] // cell_size[1] * cell_size[1],img.shape[0] // cell_size[0] * cell_size[0])
blockSize = (block_size[1] * cell_size[1],block_size[0] * cell_size[0])
blockStride = (cell_size[1],cell_size[0])
print("image shape crop by winSize (pixel): ",winSize)
print("size of 1 block (pixel): ",blockSize)
print("size of block stride (pixel):",blockStride)

hog = cv2.HOGDescriptor(_winSize=winSize,_blockSize=blockSize,_blockStride = blockStride,_cellSize = cell_size,_nbins=nbins)
n_cells = (img.shape[0] // cell_size[0],img.shape[1] // cell_size[1])
print("kich thuoc luoi o vuong: ",n_cells)

hog_feats = hog.compute(img).reshape(n_cells[1] - block_size[1]+1,n_cells[0] - block_size[0]+1,block_size[0],block_size[1],nbins).transpose((1,0,2,3,4))
print("Kich thuoc hog feature(h,w,block_size_h,block_size_w,nbins): ",hog_feats.shape)                                      

from skimage import feature
H = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")
#orientations:so bin phan chia cua  phuong gradient trong bieu do histogram
#pixels_per_cell:kich thuoc cua 1 cell(pixels)
#cells_per_block:kich thuoc cua 1 block(cells)
#block_norm:phuong phap chuan hoa block
#ham chi nhan anh 2 chieu
print('Kich thuoc hog features: ',H.shape)