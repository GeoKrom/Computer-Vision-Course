import numpy as np
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
from PIL import Image
import sys


if (len(sys.argv) != 4):
    print("Please insert the correct arguments!!")
    print("Usage: python3 <Input Image> <Output Image> <Threshold>")
    sys.exit()


# Please insert the desired Picture
I = np.array(Image.open(sys.argv[1])) # First parameter from the terminal
H = I.shape[0]
W = I.shape[1]

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
image = Image.open(sys.argv[1], 'r')
# Copy the original Image to a new one
Im2 = np.copy(image)
k = sys.argv[3]
#Function which has as input an Image and a threshold
# and as an output a procced one
def Thresholding(th, Image):
    for i in range(len(Image)):
        for j in range(len(Image[i])):
            if (Image[i][j] > th):
                Image[i][j] =  255
            else:
                Image[i][j] = 0


Thresholding(k, Im2)
print(image)
print(Im2)
plt1.figure(1)
plt1.imshow(I, cmap="gray")
plt1.show()
plt2.figure(2)
plt2.imshow(Im2, cmap="gray")
plt2.show()
# Stores the image with the name of the second parameter of the terminal
Image.fromarray(Im2.astype(np.uint8)).save(sys.argv[2])