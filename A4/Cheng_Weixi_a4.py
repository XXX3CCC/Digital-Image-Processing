import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, img_as_ubyte
from scipy.signal import convolve2d as conv2
from skimage.filters import gaussian,laplace,threshold_otsu
import scipy.ndimage
import skimage
import math
from scipy import ndimage
import cv2
from skimage.measure import find_contours




imfile = 'img_A4_P1.bmp'
I = io.imread(imfile,as_gray=True)
plt.imshow(I,cmap='jet'),plt.title('input image')
plt.show()



#PART 1
 
def part1():
    #1. Create a Laplacian-of-Gaussian Volume
    """add your code here"""
    #resource: https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian
    #resource: https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.laplace
    sp = I.shape
    h = sp[0]   #height(rows) of image
    w = sp[1]   #width(colums) of image
    #Choose 3 different sigma values in order to get the best results
    sigma1 = 3
    sigma2 = 4
    sigma3 = 5
    #With each level, we first compute the kernel size k = 2 * round(3 * sigma) + 1
    #And then use skimage.filters.gaussian followed by skimage.filters.laplace to apply the LoG filter
    #And show the results of each level 
    #level 1
    k1 = int(2 * round(3 * sigma1) + 1)
    I1 = gaussian(I, sigma = sigma1)
    I1 = laplace(I1, ksize = k1)
    plt.imshow(I1, cmap='jet')
    plt.title('Level 1')
    plt.show()
    #level 2
    k2 = int(2 * round(3 * sigma2) + 1)
    I2 = gaussian(I, sigma = sigma2)
    I2 = laplace(I2, ksize = k2)
    plt.imshow(I2, cmap='jet')
    plt.title('Level 2')
    plt.show()
    #level 3
    k3 = int(2 * round(3 * sigma3) + 1)
    I3 = gaussian(I, sigma = sigma3)
    I3 = laplace(I3, ksize = k3)
    plt.imshow(I3, cmap='jet')
    plt.title('Level 3')
    plt.show()
    #Then we need to store all these 3 levels of the volume
    #in a single numpy array whose size is h x w x 3
    #Initialize the array L
    L = np.zeros((h, w, 3), np.float64)
    L[:, :, 0] = I1
    L[:, :, 1] = I2
    L[:, :, 2] = I3

    #2. Obtain a rough estimate of blob locations
    """add your code here"""
    #resource: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.minimum_filter.html
    #By using the scipy function scipy.ndimage.filters.minimum_filter
    #The minimum value of this area is detected within the LoG amount
    min = ndimage.minimum_filter(L, size = 10)
    #the scipy function returns the actual value of the detected minimum value, not its position
    #so additional steps are required to convert its output into the desired binary image
    p = (L == min)
    #resource: https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.sum.html
    #Using np.sum to calculate the sum of the corresponding pixels in the 3 channels
    #this 3D binary image is folded into a channel image
    sum_p = np.sum(p, axis=2)
    #resource: https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html
    #Display the positions of all non-zero entries in this collapsed array overlaid on the input image in red dots using np.nonzero
    red = np.nonzero(sum_p)
    #resource: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
    #resource: https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size
    #Using plt.scatter to show the result
    plt.scatter(red[1], red[0], s = 3, c = 'r', marker = 'o')
    plt.imshow(I, cmap='jet')
    plt.title('Rough blobs detected in the image')
    plt.show()


    #3. Refine the blobs using Otsu thresholding
    """add your code here"""
    #Using skimage.filters.gaussian to get the blur image
    I_g = gaussian(I, sigma=2)
    plt.imshow(I_g, cmap='jet')
    plt.title('Blurred image')
    plt.show()

    """add your code here"""
    #resource: https://scikit-image.org/docs/dev/api/skimage.html#skimage.img_as_ubyte
    #Using skimage.img_as_ubyte convert the gaussian blurred image to 8-bit unsigned integer format
    img_as_ubyte(I_g)
    #resource: https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.threshold_otsu
    #resource: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_thresholding.html
    #Using skimage.filters.threshold_otsu to get the optimal threshold of the image
    #Delete all the minimum values in the output image of "obtain a rough estimate of spot position"
    #whose pixel value is less than the obtained threshold
    threshold = threshold_otsu(I_g)
    binary = I_g > threshold
    new = np.multiply(sum_p, binary)
    new_red = np.nonzero(new)
    #Using plt.scatter to show the result
    plt.scatter(new_red[1], new_red[0], s = 3, c = 'r', marker = 'o')
    plt.imshow(I, cmap='jet')
    plt.title('Refined blobs detected in the image')
    plt.show()


#PART 2
def part2():
    def getSmallestNeighborIndex(img, row, col):
        min_row_id = -1
        min_col_id = -1
        min_val = np.inf
        h, w = img.shape
        for row_id in range(row - 1, row + 2):
            if row_id < 0 or row_id >= h:
                continue
            for col_id in range(col - 1, col + 2):
                if col_id < 0 or col_id >= w:
                    continue
                if row_id == row and col_id == col:
                    continue
                if img[row_id, col_id] < min_val:
                    min_row_id = row_id
                    min_col_id = col_id
                    min_val = img[row_id, col_id]
        return min_row_id, min_col_id

    def getRegionalMinima(img):
        #initialize the matrix
        regional_minima = np.zeros(img.shape, dtype=np.int32)
        h, w = img.shape
        #Your code here
        #resource: https://en.wikipedia.org/wiki/Moore_neighborhood
        #Compare each pixel in the image with its 8 connected neighbors
        #Mark it as a local minimum if its value is less than all pixels
        #thereby calculating the local minimum in a given image
        num = 1
        for x in range(h):
          for y in range(w):
            min = []     #a list of minimum value
            lm = True    #a local minimum
            #The neighborhood is composed of nine cells: a central cell and the eight cells which surround it
            #NorthWest cell
            if 0 <= x - 1 and 0 <= y - 1:
              min.append(img[x - 1, y - 1])
            #North cell
            if 0 <= x - 1:
              min.append(img[x - 1, y])
            #NorthEast cell
            if 0 <= x - 1 and y + 1 < w:
              min.append(img[x - 1, y + 1])
            #West cell
            if 0 <= y - 1:
              min.append(img[x, y - 1])
            #Eest cell
            if y + 1 < w:
              min.append(img[x, y + 1])
            #SouthWest cell
            if x + 1 < h and 0 <= y - 1:
              min.append(img[x + 1, y - 1])
            #South cell
            if x + 1 < h:
              min.append(img[x + 1, y])
            #SouthEest cell
            if x + 1 < h and y + 1 < w:
              min.append(img[x + 1, y + 1])
            for i in min:
              if img[x, y] > i:
                lm = False
            if lm:
              regional_minima[x, y] = num
              num += 1
        return regional_minima



    def iterativeMinFollowing(img, markers):
        markers_copy = np.copy(markers)
        h, w = img.shape
        #a count of the number of unlabeled pixels
        n_unmarked_pix = 0
        for x in range(h):
          for y in range(w):
            #If p has been marked, leave it unchanged.
            if markers_copy[x, y] != 0:
              pass
            #Otherwise, count the number of unlabeled pixels
            else:
              n_unmarked_pix += 1
        #print(markers_copy)
        process = True
        while process == True:
            #Your code here
            #Use the minimum follow algorithm
            #to mark unmarked pixels in the marked image generated by getRegionalMinima(img)
            #implement an iterative variant of this algorithm that performs multiple passes over the image
            #Perform four steps for each pixel p in the image
            #1. If p has been marked, leave it unchanged.
            #2. If not, find the pixel with the smallest intensity value among the 8 connected neighborhoods of p.
            #3. Mark p with its label if the smallest neighbour has a non-zero label. otherwise, leave it unchanged.
            #4. Move to the next pixel and repeat steps one to three.
            for x in range(h):
              for y in range(w):
                if markers_copy[x, y] == 0:
                  min = []     #a list of minimum value
                  #The neighborhood is composed of nine cells: a central cell and the eight cells which surround it
                  #NorthWest cell
                  if 0 <= x - 1 and 0 <= y - 1:
                    min.append((x - 1, y - 1))
                  #North cell
                  if 0 <= x - 1:
                    min.append((x - 1, y))
                  #NorthEast cell
                  if 0 <= x - 1 and y + 1 < w:
                    min.append((x - 1, y + 1))
                  #West cell
                  if 0 <= y - 1:
                    min.append((x, y - 1))
                  #Eest cell
                  if y + 1 < w:
                    min.append((x, y + 1))
                  #SouthWest cell
                  if x + 1 < h and 0 <= y - 1:
                    min.append((x + 1, y - 1))
                  #South cell
                  if x + 1 < h:
                    min.append((x + 1, y))
                  #SouthEest cell
                  if x + 1 < h and y + 1 < w:
                    min.append((x + 1, y + 1))
                  #If the pixel has been marked
                  #find the pixel which has the smallest intensity value among the 8 connected neighborhoods of it
                  n = min[0]
                  #print(min)
                  for i in min:
                    if img[i] < img[n]:
                      n = i
                  #Mark the pixel with its label if the smallest neighbour has a non-zero label
                  #Leave it unchanged if the smallest neighbour does not have a non-zero label
                  if markers_copy[n] != 0:
                    markers_copy[x, y] = markers_copy[n]
                    n_unmarked_pix -= 1
                  #If the count of the number of unlabeled pixels is zero after passing
                  #the entire image has been marked as completed
                  #Otherwise, perform another pass of processing on the image
                  #This count should decrease in each successive iteration
                  if n_unmarked_pix <= 0:
                    process = False
                  #Print a count of the number of unlabeled pixels after each pass is over
                  print ('n_unmarked_pix: ', n_unmarked_pix)   
                  #print(process)  
        return markers_copy

    test_image = np.loadtxt('A4_test_image.txt')
    markers = getRegionalMinima(test_image)
    print(markers)
    labels = iterativeMinFollowing(test_image, markers)
    print(labels)

    def imreconstruct(marker, mask):
        curr_marker = (np.copy(marker)).astype(mask.dtype)
        kernel = np.ones([3, 3])
        while True:
            next_marker = cv2.dilate(curr_marker, kernel, iterations=1)
            intersection = next_marker > mask
            next_marker[intersection] = mask[intersection]
            #perform morphological reconstruction of the image marker under the image mask, and returns the reconstruction in imresonctruct
            if np.array_equal(next_marker,curr_marker):
              return curr_marker
            curr_marker = next_marker.copy()

        return curr_marker


    def imimposemin(marker, mask):
        # adapted from its namesake in MATLAB
        fm = np.copy(mask)
        fm[marker] = -np.inf
        fm[np.invert(marker)] = np.inf
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            range = float(np.max(mask) - np.min(mask))
            if range == 0:
                h = 0.1
            else:
                h = range * 0.001
        else:
            # Add 1 to integer images.
            h = 1
        fp1 = mask + h
        g = np.minimum(fp1, fm)#If marker:-inf. Else:(||grad||+h)
        return np.invert(imreconstruct(
            np.invert(fm.astype(np.uint8)), np.invert(g.astype(np.uint8))
        ).astype(np.uint8))

    sigma = 2.5
    img_name = 'img_A4_P1.bmp'
    img_rgb = io.imread(img_name).astype(np.float32)
    img_gs = skimage.color.rgb2gray(img_rgb)

    img_blurred = cv2.GaussianBlur(img_gs, (int(2 * round(3 * sigma) + 1), int(2 * round(3 * sigma) + 1)), sigma
                         )#borderType=cv2.BORDER_REPLICATE

    [img_grad_y, img_grad_x] = np.gradient(img_blurred)
    img_grad = np.square(img_grad_x) + np.square(img_grad_y)

    # refined blob locations generated generated in part 3 of lab 6
    blob_markers = np.loadtxt('A4_blob_markers.txt', dtype=np.bool, delimiter='\t')

    img_grad_min_imposed = imimposemin(blob_markers, img_grad)

    markers = getRegionalMinima(img_grad_min_imposed)
    plt.figure(0)
    plt.imshow(markers,cmap='jet')
    plt.title('markers')

    labels = iterativeMinFollowing(img_grad_min_imposed, np.copy(markers))
    plt.figure(1)
    plt.imshow(labels,cmap='jet')
    plt.title('labels')

    #contour of img_grad_min_imposed
    contours = find_contours(img_grad_min_imposed,0.8)
    contour_id = 0
    pruned_contours = []
    n_pruned_contours = 0

    fig,ax=plt.subplots()
    ax.imshow(img_grad_min_imposed, interpolation='nearest', cmap=plt.cm.gray)#
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()



if __name__ == '__main__':
    part1()
    part2()


