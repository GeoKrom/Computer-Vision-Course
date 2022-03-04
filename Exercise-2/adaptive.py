import sys
import numpy as np
from numpy import *
import math
import matplotlib.pyplot as plt
from PIL import Image
'''
 This is a script file for adaptive thresholding in combine with the otsu algorithm.
 The main difference with the simple otsu algorithm,is that we use a small window to
 compute a threshold for every intensity of the picture.

'''
# It checks for the right parameters
if len(sys.argv) != 5:
    print("Please insert the correct number of parameters!!")
    print("Usage: python3 adaptive.py <Input_Image> <Output_Image> <window_size>")
    sys.exit(1)

# Please insert the desired Picture
I = np.array(Image.open(sys.argv[2])) # Second parameter from the terminal
H = I.shape[0]
W = I.shape[1]
window_size = int(sys.argv[4]) # Fourth parameter from the terminal
print("The spartial resolution of the image is: ", H, "x", W)
# If the input image is coloured
# then we use the mean value to convert into grayscale
A = np.zeros([H,W])
I = double(I)
if len(I.shape) == 3:
    for i in range (0, H):
        for j in range(0, W):
            A[i][j] = np.mean(I[i][j], axis = -1)
else:
    for i in range(0, H):
        for j in range(0, W):
            A[i][j] = I[i][j]
image = Image.open(sys.argv[2], 'r')

# This function thresholds a picture with the best threshold and uses the Otsu Algorithm
def AdaptiveOtsuThresholder(Image):
    prob1 = np.zeros(Image)
    prob2 = np.zeros(Image)
    pixel = np.zeros(Image)
    ni = np.zeros(Image)
    pi = np.zeros(Image)
    total_meanval = 0
    objFunc = np.zeros(Image)
    thresholds = list()
    best_threshold = 0
    for i in range(0, H, window_size):
        for j in range(0, W, window_size):
            ni = np.zeros(Image)
            pi = np.zeros(Image)
            best_threshold = 0
            prob1 = np.zeros(Image)
            prob2 = np.zeros(Image)
            meanval1 = np.zeros(Image)
            meanval2 = np.zeros(Image)
            max_value = 0
            regionHeigth = 0
            regionWidth = 0
            for regH in range(i, i + window_size):
                if k > H:
                    break
                for regW in range (j, j + window_size):
                    if regW > W:
                        break
                    for k_thresh in range(0, 255):
                        pi[k_thresh] = ni[k_thresh]/(window_size**2)
                        if k_thresh == 0:
                            prob1[k_thresh] = pi[k_thresh]
                        else:
                            prob1[k_thresh] = prob1[k_thresh - 1] + pi[k_thresh]
                            prob2[k_thresh] = prob2[k_thresh] + (1 - prob1[k_thresh])
                        for k in range(0, 255):
                            if k <= k_thresh and prob1[k_thresh] != 0:
                                meanval1 += k*pi[k]/prob1[k_thresh]
                            elif k > k_thresh and prob2[k_thresh] != 0:
                                meanval2 += k*pi[k]/prob2[k_thresh]
                        total_meanval = prob1[k_thresh]*meanval1[k_thresh] + prob2[k_thresh]*meanval2[k_thresh]
                        objFunc[k_thresh] = prob1*(meanval1[k_thresh] - total_meanval)**2 + prob2[k_thresh]*(meanval2[k_thresh] - total_meanval)**2
                        if objFunc[k_thresh] > max_value:
                            max_value = objFunc[k_thresh]
                            regionHeigth = regH
                            regionWidth = regW
            thresholds.append([max_value, regionHeigth, regionWidth])
    counter = 0
    for i in range(0, H, window_size):
        for j in range(i, W, window_size):
            for k in range(i,i + window_size):
                for z in range(j,j + window_size):
                    if Image[k][z]  < thresholds[counter]:
                        Image[k][z] = 0
                    else:
                        Image[k][z] = 255
            counter+=1
    print(thresholds)

I_otsu = AdaptiveOtsuThresholder(image)
fig = plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1)
plt.imshow(I, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(I_otsu, cmap='gray')
plt.show()
# Stores the image with the name of the second parameter of the terminal
Image.fromarray(I_otsu.astype(np.uint8)).save(sys.argv[3])
