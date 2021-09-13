#LAB 2 barebones
# -*- coding: utf-8 -*
#I can run it using command 'python3 A2_submission.py'
#However, I cannot run it using command 'python A2_submission.py'


#IMPORTANT: PLEASE READ BEFORE DOING ANYTHING.
#Create your scripts in a way that just requires the TA to run it. 
#ANY script that does not display results at the first attempt,
# and/or that needs input form the TA will suffer a penalization.
#The way the script is expected to work is:
#TA types script name with the image as argument plus parameters and runs on its own
import skimage
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import filters
from skimage.morphology import disk
from skimage import io
from skimage import feature
from scipy import ndimage as ndi


#For part 1, we firstly have to set filters
#Then we need to apply different filters to every pixel in the image
#In oder to set filters, we first pad the image, which means put another circle of pixels around the image
#Then we point by point multiplication is performed on each element of the filter and image block
#After these, we sum it and specify the position of the output center pixel in the image block
#As for how to apply different filters to every pixel in the image
#We need to slide down one pixel and return to the leftmost margin and repeat block
#And to slide the filter to the right until it touches the right edge of the image
def part1():
    """add your code here"""
    #Read the grayscale image called moon.png
    img = cv2.imread('moon.png', 0)
    sp = img.shape
    H = sp[0]   #height(rows) of image
    W = sp[1]   #width(colums) of image

    #Different Filters
    kernel1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    kernel2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    kernel3 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    kernel4 = 1/9 * np.ones((3,3),np.float32)
    #Since three kernal have the same size
    Kh = kernel1.shape[0]   #height(rows) of matrix
    Kw = kernel1.shape[1]   #width(colums) of matrix
    h = int((Kh - 1) / 2)
    w = int((Kw - 1) / 2)

    #Pad. DIY
    #Create an empty array whose size is the sum of the rows
    #by the sum of the columns of the image plus the kernel
    #resource: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    img_padded = np.zeros((H + h*2, W + w*2))
    #Initialize output image to all zeros
    output1 = np.zeros_like(img_padded)
    output2 = np.zeros_like(img_padded)
    output3 = np.zeros_like(img_padded)
    output4 = np.zeros_like(img_padded)
    #print(img_padded.shape)
    #copy the pixels we need into the padding area
    #resource: https://stackoverflow.com/questions/43391205/add-padding-to-images-to-get-them-into-the-same-shape
    img_padded[h : h + H, w : w + W] = img
    #Filter the grayscale image with the following filters
    #Laplacian Filter
    for i in range(h, H-h):
      for j in range(w, W-w):
        for m in range(-h, h+1):
          for n in range(-w, w+1):
            output1[i, j] += img_padded[i+m, j+n] * kernel1[m+h, n+w]
    new1 = np.zeros((H, W))
    new1 = output1[h : h + H, w : w + W]
    #The Second Filter
    for i in range(h, H-h):
      for j in range(w, W-w):
        for m in range(-h, h+1):
          for n in range(-w, w+1):
            output2[i, j] += img_padded[i+m, j+n] * kernel2[m+h, n+w]
    new2 = np.zeros((H, W))
    new2 = output2[h : h + H, w : w + W]
    #The Third Filter
    for i in range(h, H-h):
      for j in range(w, W-w):
        for m in range(-h, h+1):
          for n in range(-w, w+1):
            output3[i, j] += img_padded[i+m, j+n] * kernel3[m+h, n+w]
    new3 = np.zeros((H, W))
    new3 = output3[h : h + H, w : w + W]
    #The Third Filter
    for i in range(h, H-h):
      for j in range(w, W-w):
        for m in range(-h, h+1):
          for n in range(-w, w+1):
            output4[i, j] += img_padded[i+m, j+n] * kernel4[m+h, n+w]
    new4 = np.zeros((H, W))
    new4 = output3[h : h + H, w : w + W]

    #Display results
    plt.imshow(img,'gray')
    plt.title('Original Image')
    plt.figure(figsize=(15, 10))
    plt.subplot(2,2,1)
    plt.imshow(new1,'gray')
    plt.title('Laplacian Filtered Image')
    plt.subplot(2,2,2)
    plt.imshow(new2,'gray')
    plt.title('Second Filtered Image')
    plt.subplot(2,2,3)
    plt.imshow(new3,'gray')
    plt.title('Third Filtered Image')
    plt.subplot(2,2,4)
    plt.imshow(new4,'gray')
    plt.title('Fourth Filtered Image')
    plt.show()



#For part2, we need to apply Median and Gaussian filters yo a noisy image
#Since we can use any scikit-image functions in this part
#I searched the internet and learned how to use scikit-image functions
def part2():
    """add your code here"""
    #Read noisy.jpg corrupted with salt and pepper noise
    img = io.imread('noisy.jpg')

    #Apply a median filter to remove the noise
    #resource: https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.median
    median_filter = filters.median(img, disk(5))

    #Apply a Gaussian filter to the same noisy image
    #resource: https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian
    Gaussian_filter = filters.gaussian(img, multichannel = True)
    #Gaussian_filter = cv2.GaussianBlur(img, (5, 5), 0)

    #Display results
    plt.figure(figsize=(10, 10))
    plt.subplot(1,3,1)
    plt.imshow(img,'gray')
    plt.title('Original Image')
    plt.subplot(1,3,2)
    plt.imshow(median_filter,'gray')
    plt.title('Median Filtered Image')
    plt.subplot(1,3,3)
    plt.imshow(Gaussian_filter,'gray')
    plt.title('Gaussian Filtered Image')
    plt.show()

    #I think median filter was more successful 
    #since the Median Filtered Image displayed without salt and pepper noise
    #the new image looks more smooth



#For part3, we have two original images
#The damaged image and the mask image
#So that we can first restores the positions of damaged pixels, which is from the mask image
#Then we use an iterative algorithm
#Smooth the damaged image by applying a Gaussian smoothing filter
#The damaged pixels in the original corrupted image are replaced with the blurred pixels in the image
#In this way we can inpaint the image
def part3():
    """add your code here"""
    #Read the damaged image and mask image
    img = cv2.imread('damage_cameraman.png')
    mask = cv2.imread('damage_mask.png')
    #repaired image = damaged image
    output = cv2.imread('damage_cameraman.png')

    sp = img.shape
    H = sp[0]   #height(rows) of image
    W = sp[1]   #width(colums) of image

    #resource: https://homepages.inf.ed.ac.uk/rbf/HIPR2/value.htm
    #resource: https://numpy.org/doc/stable/reference/generated/numpy.all.html
    #record the "bad" pixels' position
    bad_pixel = []
    for i in range(H):
      for j in range(W):
        #check if there is a good pixel
        #Typically zero is taken to be black, the damaged lines are black
        if np.all(mask[i][j]) == 0: 
          bad_pixel.append((i,j))

    #We can use an iterative algorithm
	#At every iteration, we first blurs the entire damaged image by applying a Gaussian smoothing filter
	#Since we restores damaged pixels positions before
	#we can repeat these two steps many times until all damaged pixels are infilled by smooth pixel
    for i in range(H):
      #Blurr image to fill-in voids
      #Gaussian_filter = filters.gaussian(output, multichannel = True)
      #In this part I had some trouble with smoothing
      #I tried to use skimage and other build-in functions
      #However, I found that only cv2 can solve this problem
      Gaussian_filter = cv2.GaussianBlur(output, (5, 5), 0)
      #Copy “good” pixels
      for j in bad_pixel:
        output[j] = Gaussian_filter[j]

    #display results
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title('Damaged Image')
    plt.subplot(1,2,2)
    plt.imshow(output)
    plt.title('Repaired Image')
    plt.show()




#For part4, we need to apply sobel filter to the image
#We need to get the horizontal direction derivative, vertical direction derivative and the gradient magnitude images
#Since we can use skimage.filters.sobel_v and skimage.filters.sobel_h functions
#We can easily apply these two build-in functions to get the horizontal and vertical direction derivative images
#As for the gradient magnitude image
#It can be calculated by the formula of V and H
def part4():
    """add your code here"""
	#Read the grayscale image called ex2.jpg
    img = cv2.imread('ex2.jpg', 0)

	#resource: https://scikit-image.org/docs/dev/api/skimage.filters.html
	#resource: https://en.wikipedia.org/wiki/Sobel_operator
	#Compute gradient of the image by Sobel operators
	#both the horizontal derivative and vertical derivative
    img_h = filters.sobel_h(img)   #horizontal derivative
    img_v = filters.sobel_v(img)   #vertical derivative

	#G = sqrt(Gx^2 + Gy^2)
    img_g = (img_h*img_h + img_v*img_v)**(1/2)  #gradient of the image by Sobel operators

	#Display results
    plt.figure(figsize=(15, 10))
    plt.subplot(2,2,1)
    plt.imshow(img,'gray')
    plt.title('Original Image')
	#Display the horizontal and vertical direction derivative images
    plt.subplot(2,2,2)
    plt.imshow(img_h,'gray')
    plt.title('Horizontal Image')
    plt.subplot(2,2,3)
    plt.imshow(img_v,'gray')
    plt.title('Vertical Image')
	#Display the gradient magnitude image
    plt.subplot(2,2,4)
    plt.imshow(img_g,'gray')
    plt.title('Gradient Magnitude  Image')
    plt.show()



#For part5, we need to apply Canny edge detection
#There are four steps of the Canny edge detection
#Step 1: Smooth the image with Gaussian filter
#Step 2: Calculate the gradient amplitude and direction with the finite difference of first-order partial derivative
#Step 3: Suppress the gradient amplitude with non maximum value
#Step 4: Detect and connect the edge with double threshold algorithm
#We need to apply these steps and research the effect of threshold values and sigma value with different variables
def part5():
    """add your code here"""
    #Read the grayscale image called ex2.jpg
    img = io.imread('ex2.jpg', as_gray = True)

    #Display the original image
    plt.imshow(img,'gray')
    plt.title('Original Image')
    plt.show()

    #Step 1
    #Apply a Gaussian filter to the same noisy image
    #img = cv2.GaussianBlur(img, (5, 5), 0)
    img = filters.gaussian(img, multichannel = True)
    #img = ndi.gaussian_filter(img, 4)

    #Display the smooth image
    plt.imshow(img,'gray')
    plt.title('Smooth Image')
    plt.show()

    #Step 2
    img_h = filters.sobel_h(img)   #get horizontal derivative
    img_v = filters.sobel_v(img)   #get vertical derivative
    #get the gradient of the image by Sobel operators
    #G = sqrt(Gx^2 + Gy^2)   --->   G = |Gx| + |Gy|
    #img = abs(img_h) + abs(img_v)
    img_g = (img_h*img_h + img_v*img_v)**(1/2)

    #Display the gradient magnitude image
    plt.imshow(img,'gray')
    plt.title('gradient Image')
    plt.show()

    #Convert the image to 8-bit unsigned integer format
    img = skimage.img_as_ubyte(img)

    #resource: https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.canny
    edge1 = feature.canny(img, sigma = 1.0, low_threshold = 25)
    edge2 = feature.canny(img, sigma = 1.0, low_threshold = 50)
    edge3 = feature.canny(img, sigma = 1.0, high_threshold = 150)
    edge4 = feature.canny(img, sigma = 1.0, high_threshold = 200)

    #Display results
    #In order to find out the effect of threshold values
    #We fix sigma = 1.0
    #Then we plot the following 4 figures together and see their difference
    #1. low threshold = 25
    plt.figure(figsize=(15, 10))
    plt.subplot(2,2,1)
    plt.imshow(edge1,'gray')
    plt.title('the effect of threshold values: first Image')
    #2. low threshold = 50
    plt.subplot(2,2,2)
    plt.imshow(edge2,'gray')
    plt.title('the effect of threshold values: second Image')
    #3. high threshold = 150
    plt.subplot(2,2,3)
    plt.imshow(edge3,'gray')
    plt.title('the effect of threshold values: third Image')
    #4. high threshold = 200
    plt.subplot(2,2,4)
    plt.imshow(edge4,'gray')
    plt.title('the effect of threshold values: fourth Image')
    plt.show()

    edge1 = feature.canny(img, sigma = 1.0, low_threshold = 50, high_threshold = 150)
    edge2 = feature.canny(img, sigma = 1.5, low_threshold = 50, high_threshold = 150)
    edge3 = feature.canny(img, sigma = 2.0, low_threshold = 50, high_threshold = 150)
    edge4 = feature.canny(img, sigma = 2.5, low_threshold = 50, high_threshold = 150)

    #Display results
    #In order to find out the effect of sigma value
    #we fix low_threshold=50 and high_threshold=150
    #Then we plot the following 4 figures together and see their difference :
    #1. sigma =  1.0
    plt.figure(figsize=(15, 10))
    plt.subplot(2,2,1)
    plt.imshow(edge1,'gray')
    plt.title('the effect of sigma value: first Image')
    #2. sigma = 1.5
    plt.subplot(2,2,2)
    plt.imshow(edge2,'gray')
    plt.title('the effect of sigma value: second Image')
    #3. sigma = 2.0
    plt.subplot(2,2,3)
    plt.imshow(edge3,'gray')
    plt.title('the effect of sigma value: third Image')
    #4. sigma = 2.5
    plt.subplot(2,2,4)
    plt.imshow(edge4,'gray')
    plt.title('the effect of sigma value: fourth Image')
    plt.show()

#resource: http://www.kerrywong.com/2009/05/07/canny-edge-detection-auto-thresholding/
#resource: https://dsp.stackexchange.com/questions/1733/could-you-describe-the-effects-for-varying-different-parameters-of-a-canny-edge
#For the effect of threshold values
#We can find that if the gap between low threshold value and high threshold value is small
#the continuity of the resulting edges will be reduced
#In this way there will be more fractions
#When the gap between low threshold value and high threshold value increases, we will have more single-line edges
#Otherwise the line segments will be too many and complicated
#As for the effect of sigma value
#We can find that the larger the sigma value, the less obvious the edge
#And when the sigma value increases, the edge will become smoother and the noisy edges will disappear




if __name__ == '__main__':
	part1()
	part2()
	part3()
	part4()
	part5()



