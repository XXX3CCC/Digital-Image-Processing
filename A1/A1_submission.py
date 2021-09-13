import os
os.chdir(os.path.dirname(__file__))

import math
import cv2
from skimage import io
from skimage import exposure
import numpy as np
from matplotlib import pyplot as plt

def part1_histogram_compute():
  print('PART 1')

  #Read the grayscale image called test.jpg
  img = cv2.imread('test.jpg', 0)

  sp = img.shape
  h = sp[0]   #height(rows) of image
  w = sp[1]   #width(colums) of image

  #Use our own function to compute image histogram
  hist_my = np.zeros(256)
  for i in range (0, h):
    for j in range (0, w):
      hist_my[img[i][j]] += 1

  #Use Skimage function to compute image histogram
  hist_S, bins_S = exposure.histogram(img, source_range='dtype')

  #Use Numpy function to compute image histogram
  hist_N, bins_N = np.histogram(img, 256, [0, 256])

  #Plot both histograms side by side
  #to show that they are identical
  plt.subplots_adjust(wspace=1)   #adjust width of the subplots
  #Plot the histogram with our own method
  plt.subplot(1,3,1)
  plt.plot(hist_my)
  plt.title('Our Own Histogram')
  plt.xlim([0,256])   #Set the x limit of the current axis to (0, 256)

  #Plot the histogram with Skimage method
  plt.subplot(1,3,2)
  plt.plot(hist_S)
  plt.title("Skimage Histogram")
  plt.xlim([0,256])

  #Plot the histogram with Numpy method
  plt.subplot(1,3,3)
  plt.plot(hist_N)
  plt.title("Numpy Histogram")
  plt.xlim([0,256])

  plt.show()



def part2_histogram_equalization():
  print('\nPART 2')

  #Read the grayscale image called test.jpg
  img = cv2.imread('test.jpg', 0)
  sp = img.shape
  h = sp[0]   #height(rows) of image
  w = sp[1]   #width(colums) of image
  K = 256     #Intensity level

  #Compute the histogram
  hist, bins = np.histogram(img.flatten(), 256, [0,256])
  
  #Compute the cumulative histogram
  cdf = np.cumsum(hist)

  #Histogram equalization
  #source: https://en.wikipedia.org/wiki/Histogram_equalization
  cdf_m = np.ma.masked_equal(cdf, 0)
  cdf_m = (cdf_m - np.min(cdf_m)) * (K - 1) / (np.max(cdf_m) - np.min(cdf_m))
  cdf = np.ma.filled(cdf_m, 0).astype('uint8')

  #Get the image after histogram equalization
  img_E = cdf[img]

  #Compute its histogram
  hist_E, bins_E = np.histogram(img_E.flatten(), 256, [0,256])

  #Plot the original image and its histogram
  plt.figure(figsize=(15, 10))
  plt.subplots_adjust(hspace=.5)
  plt.subplots_adjust(wspace=.3)
  plt.subplot(2,2,1)
  plt.imshow(img, 'gray')
  plt.title('Original Image')
  plt.subplot(2,2,2)
  plt.plot(hist)
  plt.title('Original Histogram')
  plt.xlim([0,256])

  #Plot the image after histogram equalization and its histogram
  plt.subplot(2,2,3)
  plt.imshow(img_E, 'gray')
  plt.title('New Image')
  plt.subplot(2,2,4)
  plt.plot(hist_E)
  plt.title('Histogram After Equalization')
  plt.xlim([0,256])

  plt.show()



def part3_histogram_comparing():
  print('\nPART 3')

  #Read two grayscale images
  img_day = cv2.imread('day.jpg', 0)
  img_night = cv2.imread('night.jpg', 0)

  #Use Skimage function to compute image histogram
  hist_day, bins_day = exposure.histogram(img_day, source_range='dtype')
  hist_night, bins_night = exposure.histogram(img_night, source_range='dtype')

  h_day = img_day.shape[0]   #height(rows) of image
  w_day = img_day.shape[1]   #width(colums) of image
  h_night = img_night.shape[0]
  w_night = img_night.shape[1]

  #Calculate the MNs
  MN_d = h_day * w_day
  MN_n = h_night * w_night

  #Calculate the Bhattacharyya Coefficient of the two histograms
  #Since hist_day_norm and hist_night_norm are two normalized histograms
  #Bhattacharyya Coefficient is defined as BC(p,q) = sum(p(i)*q(i))
  BC = 0
  for i in range(256):
    #Normalize two histograms
    #Normalized version is defined as p(i) = h(i)/sum(h(i)) = h(i)/MN
    hist_day_norm = hist_day[i] / MN_d
    hist_night_norm = hist_night[i] / MN_n
    BC += math.sqrt((hist_day_norm * hist_night_norm))

  #Plot two grayscale images and their histograms
  plt.figure(figsize=(15, 10))
  plt.subplots_adjust(hspace=.5)
  plt.subplots_adjust(wspace=.3)
  plt.subplot(2,2,1)
  plt.imshow(img_day,'gray')
  plt.title("Day Image")
  plt.subplot(2,2,2)
  plt.plot(hist_day)
  plt.title('Day Histogram')
  plt.subplot(2,2,3)
  plt.imshow(img_night,'gray')
  plt.title("Night Image")
  plt.subplot(2,2,4)
  plt.plot(hist_night)
  plt.title("Night Histogram")
  plt.show()

  #print the Bhattacharyya Coefficient of the two histograms
  print("Bhattacharyya Coefficient of the two histograms is", BC)



def part4_histogram_matching():
  #Inputs: 
  #(1)Input image: l
  #(2)Gray scale range:{0,...,K-1}
  #(3)Reference histogram
  #Outputs:
  #(1)Output image J
  print('\nPART4')

  #Read two grayscale images
  img_day = cv2.imread('day.jpg', 0)
  img_night = cv2.imread('night.jpg', 0) 

  #Use Skimage function to compute image histogram
  hist_day, bins_day = exposure.histogram(img_day, nbins = 256, source_range='dtype')
  hist_night, bins_night = exposure.histogram(img_night, nbins = 256, source_range='dtype')

  h_day = img_day.shape[0]   #height(rows) of image
  w_day = img_day.shape[1]   #width(colums) of image
  h_night = img_night.shape[0]
  w_night = img_night.shape[1]

  #Normalize two histograms
  #Compute PA(a) in order to get normalized cumulative input histogram
  cdf_d = []
  cdf_d.append(hist_day[0])
  for i in range(1, 256):
    cdf_d.append(cdf_d[i - 1] + hist_day[i])

  #Compute PR(a) in order to get normalized cumulative reference histogram
  cdf_n = []
  cdf_n.append(hist_night[0])
  for i in range(1, 256):
    cdf_n.append(cdf_n[i - 1] + hist_night[i])

  #Copy the day image
  img_match = np.copy(img_day)

  #Compute mapping
  for i in range(h_day):
    for j in range(w_day):
      x = img_day[i][j]
      y = 0
      while cdf_d[x] > cdf_n[y]:
        y += 1
      img_match[i][j] = y

  #read the original day image
  img_day = io.imread('day.jpg')
  img_night = io.imread('night.jpg') 
  img = io.imread('day.jpg')

  #split the image into red, green and blue
  #and repeat histogram matching to each one

  #(1)RED LAYER
  #Use Skimage function to compute image histogram
  hist_day, bins_day = exposure.histogram(img_day[:,:,0], nbins = 256, source_range='dtype')
  hist_night, bins_night = exposure.histogram(img_night[:,:,0], nbins = 256, source_range='dtype')

  h_day = img_day.shape[0]   #height(rows) of image
  w_day = img_day.shape[1]   #width(colums) of image
  h_night = img_night.shape[0]
  w_night = img_night.shape[1]

  #Normalize two histograms
  #Compute PA(a) in order to get normalized cumulative input histogram
  cdf_d = []
  cdf_d.append(hist_day[0])
  for i in range(1, 256):
    cdf_d.append(cdf_d[i - 1] + hist_day[i])

  #Compute PR(a) in order to get normalized cumulative reference histogram
  cdf_n = []
  cdf_n.append(hist_night[0])
  for i in range(1, 256):
    cdf_n.append(cdf_n[i - 1] + hist_night[i])

  #Compute mapping
  for i in range(h_day):
    for j in range(w_day):
      x = img[i][j][0]
      y = 0
      while cdf_d[x] > cdf_n[y]:
        y += 1
      img[i][j][0] = y

  #(2)GREEN LAYER
  #Use Skimage function to compute image histogram
  hist_day, bins_day = exposure.histogram(img_day[:,:,1], nbins = 256, source_range='dtype')
  hist_night, bins_night = exposure.histogram(img_night[:,:,1], nbins = 256, source_range='dtype')

  h_day = img_day.shape[0]   #height(rows) of image
  w_day = img_day.shape[1]   #width(colums) of image
  h_night = img_night.shape[0]
  w_night = img_night.shape[1]

  #Normalize two histograms
  #Compute PA(a) in order to get normalized cumulative input histogram
  cdf_d = []
  cdf_d.append(hist_day[0])
  for i in range(1, 256):
    cdf_d.append(cdf_d[i - 1] + hist_day[i])

  #Compute PR(a) in order to get normalized cumulative reference histogram
  cdf_n = []
  cdf_n.append(hist_night[0])
  for i in range(1, 256):
    cdf_n.append(cdf_n[i - 1] + hist_night[i])

  #Compute mapping
  for i in range(h_day):
    for j in range(w_day):
      x = img[i][j][1]
      y = 0
      while cdf_d[x] > cdf_n[y]:
        y += 1
      img[i][j][1] = y

  #(3)BLUE LAYER
  #Use Skimage function to compute image histogram
  hist_day, bins_day = exposure.histogram(img_day[:,:,2], nbins = 256, source_range='dtype')
  hist_night, bins_night = exposure.histogram(img_night[:,:,2], nbins = 256, source_range='dtype')

  h_day = img_day.shape[0]   #height(rows) of image
  w_day = img_day.shape[1]   #width(colums) of image
  h_night = img_night.shape[0]
  w_night = img_night.shape[1]

  #Normalize two histograms
  #Compute PA(a) in order to get normalized cumulative input histogram
  cdf_d = []
  cdf_d.append(hist_day[0])
  for i in range(1, 256):
    cdf_d.append(cdf_d[i - 1] + hist_day[i])

  #Compute PR(a) in order to get normalized cumulative reference histogram
  cdf_n = []
  cdf_n.append(hist_night[0])
  for i in range(1, 256):
    cdf_n.append(cdf_n[i - 1] + hist_night[i])

  #Compute mapping
  for i in range(h_day):
    for j in range(w_day):
      x = img[i][j][2]
      y = 0
      while cdf_d[x] > cdf_n[y]:
        y += 1
      img[i][j][2] = y


  print('(a)')
  #Show the grayscale day, night and matched day images side by side
  plt.figure(figsize=(20, 10))
  plt.subplot(1,3,1)
  plt.imshow(img_day,'gray')
  plt.title("Day Image")

  plt.subplot(1,3,2)
  plt.imshow(img_night,'gray')
  plt.title("Night Image")

  plt.subplot(1,3,3)
  plt.imshow(img_match,'gray')
  plt.title("Matched Day Image")

  plt.show()

  print('(b)')
  #Show the day, night and matched day RGB images side by side
  plt.figure(figsize=(20, 10))
  plt.subplot(1,3,1)
  plt.imshow(cv2.cvtColor(cv2.imread('day.jpg'), cv2.COLOR_BGR2RGB))
  plt.title("Day Image")

  plt.subplot(1,3,2)
  plt.imshow(cv2.cvtColor(cv2.imread('night.jpg'), cv2.COLOR_BGR2RGB))
  plt.title("Night Image")

  plt.subplot(1,3,3)
  plt.imshow(img)
  plt.title("RGB Matched Day Image")

  plt.show()

  

if __name__ == '__main__':
    part1_histogram_compute()
    part2_histogram_equalization()
    part3_histogram_comparing()
    part4_histogram_matching()
