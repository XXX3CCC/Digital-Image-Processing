# Import libraries
import numpy as np
import math
from matplotlib import pyplot as plt
import os
from sklearn.cluster import KMeans
from scipy import spatial
from skimage import io, color, img_as_float
from math import floor
from skimage.transform import warp
from skimage.transform import AffineTransform
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from skimage.transform import ProjectiveTransform
from skimage.measure import ransac
from skimage.color import gray2rgb
from skimage.color import rgb2gray
from skimage import exposure
import skimage

#Part1
def part1():
    filename_Grayimage = 'PeppersBayerGray.bmp'
    filename_gridB = 'gridB.bmp'
    filename_gridR = 'gridR.bmp'
    filename_gridG = 'gridG.bmp'

    # part I
    img = io.imread(filename_Grayimage, as_gray =True)

    h,w = img.shape

    # our final image will be a 3 dimentional image with 3 channel
    rgb = np.zeros((h,w,3),np.uint8);

    # reconstruction of the green channel IG

    IG = np.copy(img) # copy the image into each channel

    for row in range(0,h,4): # loop step is 4 since our mask size is 4.
        for col in range(0,w,4): # loop step is 4 since our mask size is 4.
        
        #B =（A + C）/ 2
            IG[row,col+1]=(int(img[row,col])+int(img[row,col+2]))/2

        #D = (C + H) / 2
            IG[row,col+3]=(int(img[row,col+2])+int(img[row+1,col+3]))/2

        #E =（A + I）/ 2
            IG[row+1,col]=(int(img[row,col])+int(img[row+2,col]))/2

        #G =（F + C + H + K）/ 4
            IG[row+1,col+2]=(int(img[row+1,col+1])+int(img[row,col+2])+int(img[row+1,col+3])+int(img[row+2,col+2]))/4

        #J = (I + F + K + N) / 4
            IG[row+2,col+1]=(int(img[row+2,col])+int(img[row+1,col+1])+int(img[row+2,col+2])+int(img[row+3,col+1]))/4

        #L = (H + P) / 2
            IG[row+2,col+3]=(int(img[row+1,col+3])+int(img[row+3,col+3]))/2

        #M = (I + N) / 2
            IG[row+3,col]= (int(img[row+2,col])+int(img[row+3,col+1]))/2

        #O = (N + P) / 2
            IG[row+3,col+2]=(int(img[row+3,col+1])+int(img[row+3,col+3]))/2


    # reconstruction of the red channel IR

    IR = np.copy(img) # copy the image into each channel

    for row in range(0,h,4): # loop step is 4 since our mask size is 4.
        for col in range(0,w,4): # loop step is 4 since our mask size is 4.

        #As for the red channel, we know that the first column(col) and the last row(row+3) are entirely empty
        #So we need to fill them in by respectively copying the second column(col+1) and the second last row(row+2)

        #F =（B + J）/ 2
            IR[row+1,col+1]=(int(img[row,col+1])+int(img[row+2,col+1]))/2

        #C =（B + D）/ 2 
            IR[row,col+2]=(int(img[row,col+1])+int(img[row,col+3]))/2

        #H = (D + L) / 2
            IR[row+1,col+3]=(int(img[row,col+3])+int(img[row+2,col+3]))/2

        #K = (J + L) / 2
            IR[row+2,col+2]=(int(img[row+2,col+1])+int(img[row+2,col+3]))/2

        #G =（B + D + L + J）/ 4
            IR[row+1,col+2]=(int(img[row,col+1])+int(img[row,col+3])+int(img[row+2,col+3])+int(img[row+2,col+1]))/4

        #A = B
            IR[row,col]=IR[row,col+1]

        #E = F
            IR[row+1,col]=IR[row+1,col+1]

        #I = J
            IR[row+2,col]=IR[row+2,col+1]

        #N = J
            IR[row+3,col+1]=IR[row+2,col+1]

        #O = K
            IR[row+3,col+2]=IR[row+2,col+2]

        #P = L
            IR[row+3,col+3]=IR[row+2,col+3]

        #M = N = J
            IR[row+3,col]=IR[row+2,col+1]


    # reconstruction of the blue channel IB
    IB = np.copy(img) # copy the image into each channel

    for row in range(0,h,4): # loop step is 4 since our mask size is 4.
        for col in range(0,w,4): # loop step is 4 since our mask size is 4.

        #As for the blue channel, we know that the last column(col+3) and the first row(row) are entirely empty. 
        #So we need to fill them in by respectively copying the second row(row+1) and the last column(col+3)

        #F = (E + G) / 2
            IB[row+1,col+1]=(int(img[row+1,col])+int(img[row+1,col+2]))/2

        #I =（E + M）/ 2
            IB[row+2,col]=(int(img[row+1,col])+int(img[row+3,col]))/2

        #N = (M + O) / 2
            IB[row+3,col+1]=(int(img[row+3,col])+int(img[row+3,col+2]))/2

        #K = (G + O) / 2
            IB[row+2,col+2]=(int(img[row+1,col+2])+int(img[row+3,col+2]))/2

        #J =（E + G + O + M）/ 4
            IB[row+2,col+1]=(int(img[row+1,col])+int(img[row+1,col+2])+int(img[row+3,col])+int(img[row+3,col+2]))/4

        #A = E
            IB[row,col]=IB[row+1,col]

        #B = F
            IB[row,col+1]=IB[row+1,col+1]

        #C = G
            IB[row,col+2]=IB[row+1,col+2]

        #H = G
            IB[row+1,col+3]=IB[row+1,col+2]

        #L = K
            IB[row+2,col+3]=IB[row+2,col+2]

        #P = O
            IB[row+3,col+3]=IB[row+3,col+2]

        #D = H
            IB[row,col+3]=IB[row+1,col+3]


    # merge the channels
    rgb[:,:,0]=IR
    rgb[:,:,1]=IG
    rgb[:,:,2]=IB

    #show the result
    plt.imshow(rgb),plt.title('rgb')
    plt.show()


#Part2
# Finds the closest colour in the palette using kd-tree.
def nearest(palette, colour):
    dist, i = palette.query(colour)
    return palette.data[i]

# Make a kd-tree palette from the provided list of colours
def makePalette(colours):
    #print(colours)
    return spatial.KDTree(colours)

# Dynamically calculates and N-colour palette for the given image
# Uses the KMeans clustering algorithm to determine the best colours
# Returns a kd-tree palette with those colours
def findPalette(image, nColours):
    #your code
    #resource: https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
    #resource: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    #we reshape the image in order to let it be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    #cluster the pixel intensities, in order to determine the best colors
    #fit the data
    clt = KMeans(nColours).fit(image)
    #Kmeans centers
    colours = clt.cluster_centers_
    return makePalette(colours)

def FloydSteinbergDitherColor(image, palette):
#***** The following pseudo-code is grabbed from Wikipedia: https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering.  
#   for each y from top to bottom ==>(height)
#    for each x from left to right ==> (width)
#       oldpixel  := pixel[x][y]
#       newpixel  := nearest(oldpixel) # Determine the new colour for the current pixel
#       pixel[x][y]  := newpixel 
#       quant_error  := oldpixel - newpixel
#       pixel[x + 1][y    ] := pixel[x + 1][y    ] + quant_error * 7 / 16
#       pixel[x - 1][y + 1] := pixel[x - 1][y + 1] + quant_error * 3 / 16
#       pixel[x    ][y + 1] := pixel[x    ][y + 1] + quant_error * 5 / 16
#       pixel[x + 1][y + 1] := pixel[x + 1][y + 1] + quant_error * 1 / 16

    #get the height and the width of the image
    h,w = image.shape[:2]

    #copy the original image
    pixel = np.copy(image)

    #pseudo-code from Wikipedia: https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering
    for y in range(h - 1):
      for x in range(w - 1):
          oldpixel = image[x][y]
          newpixel = nearest(palette, oldpixel)
          pixel[x][y] = newpixel
          quant_error = oldpixel - newpixel
          #quant_error = [oldpixel[0] - newpixel[0], oldpixel[1] - newpixel[1], oldpixel[2] - newpixel[2]]
          pixel[x + 1][y    ] = pixel[x + 1][y    ] + quant_error * 7 / 16
          pixel[x - 1][y + 1] = pixel[x - 1][y + 1] + quant_error * 3 / 16
          pixel[x    ][y + 1] = pixel[x    ][y + 1] + quant_error * 5 / 16
          pixel[x + 1][y + 1] = pixel[x + 1][y + 1] + quant_error * 1 / 16

    return pixel

def part2():

    nColours = 8 # The number colours: change to generate a dynamic palette

    imfile = 'lena.png'
    
    image = io.imread(imfile)

    # Strip the alpha channel if it exists
    image = image[:,:,:3]

    # Convert the image from 8bits per channel to floats in each channel for precision
    image = img_as_float(image)

    # Dynamically generate an N colour palette for the given image
    palette = findPalette(image, nColours)
    colours = palette.data
    
    colours = img_as_float([colours.astype(np.ubyte)])[0]
    
    img = FloydSteinbergDitherColor(image, palette)

    #show the result
    plt.imshow(img)
    plt.show()

#Part3
def part3():
    image1 = io.imread('lab5_img.jpeg')

    #get the height, width and channel of the image1
    h, w, c = image1.shape

    #resource: https://en.wikipedia.org/wiki/Affine_transformation
    #Rotation transformation matrix(90 degrees clockwise)
    #Rotate = [[cosx, -sinx, 0], [sinx, cosx, 0], [0, 0, 1]], x = -90 degrees
    T_r = np.array([[0, 1, 0],[-1, 0, 0],[0, 0, 1]])
    #Scale transformation matrix(by two)
    #Scale = [[c_x, 0, 0], [c_y, 0, 0], [0, 0, 1]]
    T_s = np.array([[2, 0, 0],[0, 2, 0],[0, 0, 1]])

    ##Initialize output image to all zeros
    image2 = np.zeros((h * 2, w * 2, c), dtype = np.uint8)

    #Combine the transformations
    #Affine transformations are essentially linear transformations
    T_c = T_r.dot(T_s)

    #resource: https://stackoverflow.com/questions/55962521/rotate-image-in-affine-transformation
    #The combined transformation is applied to the spatial domain of the image data
    for i in range(h - 1): 
      for j in range(w - 1):
        curr_pixel = [i, j, 1]
        temp = image1[i, j, :]
        ii, jj, cc = T_c.dot(curr_pixel)
        image2[ii, jj, :] = temp

    #resource: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.AffineTransform
    #resource: https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.warp
    #Using the inverse function of the transformation matrix T
    image3 = np.zeros((h * 2, w * 2, c), dtype = np.uint8)
    tform = AffineTransform(scale = (2, 2), rotation = np.deg2rad(90))
    image3 = warp(image1, tform.inverse, output_shape = image2.shape, mode = 'wrap')

    #show results
    plt.imshow(image1,cmap='gray')
    plt.show()

    plt.imshow(image2,cmap='gray')
    plt.show()

    plt.imshow(image3,cmap='gray')
    plt.show()



#Part4
#Problem: every time the output is different image.
def part4():
    #read two images
    image0 = io.imread('im1.jpg', True)
    image1 = io.imread('im2.jpg', True)

    plt.imshow(image0,cmap='gray')
    plt.show()
    plt.imshow(image1,cmap='gray')
    plt.show()
    #Feature detection and matching

    # Detect local features with  ORB in the two images image0 and image1
    # Initiate ORB detector
    # your code #
    # resource: https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_orb.html
    # resource: https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.ORB
    # why n_keypoints = 500? when I tried n_keypoints = 200 or 1000, evertime I plot it can get different images
    descriptor_extractor = ORB(n_keypoints = 500)

    # Find the keypoints and descriptors
    # your code #
    descriptor_extractor.detect_and_extract(image0)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors
    descriptor_extractor.detect_and_extract(image1)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    # initialize Brute-Force matcher and exclude outliers. See match descriptor function.
    # your code #
    matches12 = match_descriptors(descriptors1, descriptors2, max_distance = 10, cross_check=True)

    # Compute homography matrix using ransac and ProjectiveTransform
    # your code #
    # resource: https://scikit-image.org/docs/dev/auto_examples/transform/plot_matching.html
    # resource: https://github.com/scikit-image/scikit-image-paper/blob/main/skimage/pano.txt
    # Select keypoints from the source and target
    # robustly estimate affine transform model with RANSAC
    src = keypoints2[matches12[:, 1]][:, ::-1]
    dst = keypoints1[matches12[:, 0]][:, ::-1]
    model_robust, inliers = ransac((src, dst), ProjectiveTransform, min_samples=3, residual_threshold=2)
    inliers == False
    outliers = inliers

    #Warping
    #Next, we produce the panorama itself. The first step is to find the shape of the output image by considering the extents of all warped images.
    r, c = image1.shape[:2]

    # Note that transformations take coordinates in
    # (x, y) format, not (row, column), in order to be
    # consistent with most literature.
    corners = np.array([[0, 0],
                        [0, r],
                        [c, 0],
                        [c, r]])

    # Warp the image corners to their new positions.
    warped_corners = model_robust(corners)

    # Find the extents of both the reference image and
    # the warped target image.
    all_corners = np.vstack((warped_corners, corners))

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)

    output_shape = (corner_max - corner_min)
    output_shape = np.ceil(output_shape[::-1])

    #The images are now warped according to the estimated transformation model.

    #A shift is added to ensure that both images are visible in their entirety. Note that warp takes the inverse mapping as input.

    from skimage.exposure import rescale_intensity
    from skimage.transform import warp
    from skimage.transform import SimilarityTransform

    offset = SimilarityTransform(translation=-corner_min)

    image0_ = warp(image0, offset.inverse,
                   output_shape=output_shape)

    image1_ = warp(image1, (model_robust + offset).inverse,
                   output_shape=output_shape)

    #An alpha channel is added to the warped images before merging them into a single image:

    def add_alpha(image, background=-1):
        """Add an alpha layer to the image.

        The alpha layer is set to 1 for foreground
        and 0 for background.
        """
        rgb = gray2rgb(image)
        alpha = (image != background)
        return np.dstack((rgb, alpha))


    #add alpha to the image0 and image1
    #your code
    img0 = add_alpha(image0_)
    img1 = add_alpha(image1_)

    #merge the alpha added image
    #your code
    merged = (img0 + img1)

    alpha = merged[..., 3]
    merged /= np.maximum(alpha, 1)[..., np.newaxis]
    # The summed alpha layers give us an indication of
    # how many images were combined to make up each
    # pixel.  Divide by the number of images to get
    # an average.

    #show and save the output image as '/content/gdrive/My Drive/CMPUT 206 Wi19/Lab5_Files/imgOut.png'
    #your code
    plt.imshow(merged, cmap='gray')
    plt.show()
    #path = '/content/gdrive/My Drive/CMPUT 206 Wi19/Lab5_Files/imgOut.png'
    #io.imsave(path, merged)


if __name__ == '__main__':
    part1()
    part2()
    part3()
    part4()



