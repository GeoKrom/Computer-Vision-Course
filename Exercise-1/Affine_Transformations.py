import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def affTransform(im, a1, a2, a3, a4, a5, a6):
    newIm = np.zeros(im.shape)

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            
            # Start from the center of the image and apply affine transformation
            x = i - im.shape[0]/2
            y = j - im.shape[1]/2
            
            aff_x = a1*x + a2*y + a3
            aff_y = a4*x + a5*y + a6

            # Map new position and round it up
            new_x = round(aff_x + im.shape[0]/2)
            new_y = round(aff_y + im.shape[1]/2)

            # Check if new positions are inside of image range and store the values to the output image
            if new_x in range(0,im.shape[0]) and new_y in range(0,im.shape[1]):
                newIm[i][j] = im[new_x][new_y]
    return newIm

if (len(sys.argv) != 10):
    print("Please insert the correct arguments!!")
    print("Usage: python3 Affine_Transformations.py <Input_Image> <Output_Image> <a1> <a2> <a3> <a4> <a5> <a6>")
    sys.exit()

# Open image as an array
image = Image.open(sys.argv[2], 'r')

#Plotting image
plt.imshow(image, cmap="gray")
plt.title("Image before Transformation")
plt.show()

'''
    In general there are six types of affine transformation:
        1. Indentity Transformation   {a1 = 1, a2 = 0, a3 = 0, a4 = 0, a5 = 1, a6 = 0}
        2. Translation Transformation {a1 = 1, a2 = 0, a3 = T_x, a4 = 0, a5 = 1, a6 = T_y}
        3. Scale Transformation       {a1 = S_x, a2 = 0, a3 = 0, a4 = 0, a5 = S_y, a6 = 0}
        4. Rotational Transformation  {a1 = cos(theta), a2 = -sin(theta), a3 = 0, a4 = sin(theta), a5 = cos(theta), a6 = 0}
        5. Shearing Transformation    {a1 = 1, a2 = a, a3 = 0, a4 = 0, a5 = 1, a6 = 0} or {a1 = 1, a2 = 0, a3 = 0, a4 = b, a5 = 1, a6 = 0}
        6. All the above together

                [a1 a2 a3]
    Taffine =   |a4 a5 a6|
                [0  0   1]
    
    Transformed_Image = Taffine @ Image
'''
# Transformation
TfIm = affTransform(image, float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9]))

#Plotting image
plt.imshow(TfIm, cmap="gray")
plt.title("Image after Transformation")
plt.show()

# Stores the image with the name of the second parameter of the terminal
Image.fromarray(TfIm.astype(np.uint8)).save(sys.argv[3])
