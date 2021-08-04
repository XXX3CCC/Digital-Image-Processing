from skimage import io
import matplotlib.pyplot as plt
import numpy as np

# read input images
I1 = io.imread("apple.jpg")
I2 = io.imread("orange.jpg")
M = io.imread("mask512.jpg")

# create a new figure to display the images - 15 x 5 inches in size with default 100 dpi --> 1500 x 500 pixels
fig = plt.figure(figsize=(15, 5))
# split up the figure into a 1 x 3 grid of sub-figures
plt.subplot(131)
#  display apple in figure 1 from left and label it as such
plt.imshow(I1)
plt.title("An apple")
#  display orange in figure 2 from left
plt.subplot(132)
plt.imshow(I2)
plt.title("An orange")
#  display Blending mask in figure 3 from left
plt.subplot(133)
# use grayscale colormap for the single channel mask image - colormap is ignored for RGB images above so not specified
# note that io.imread by default reads even single channel images as 3 channel arrays
# so M is actually 512x512x3 with all three channels containing the same pixel values
# as a result, it is treated as an RGB image by plt.imshow and the colormap is ignored but it would matter if the
# image was read as single channel by passing as_gray=1 to io.imread
plt.imshow(M,
           cmap="gray"
           )
plt.title("Blending mask")
# show the figure and pause execution till it is closed by user
plt.show()

from skimage.transform import pyramids

# construct Gaussian pyramids for apple and orange images containing 6 layers with each layer being half the size of
# the previous one, i.e. 512x512, 256x256, 128x128, 64x64, 32x32 and 16x16 from base to top of pyramid
# max_layer takes 0-based numbers so max_layer=5 builds 6 layers - 0, 1, 2, 3, 4, 5
# source: https://scikit-image.org/docs/0.8.0/api/skimage.transform.pyramids.html#pyramid-gaussian
# multichannel mode is enabled since these are RGB images
Apple_Gaussian = pyramids.pyramid_gaussian(I1, max_layer=5, downscale=2, multichannel=True)
Orange_Gaussian = pyramids.pyramid_gaussian(I2, max_layer=5, downscale=2, multichannel=True)
# number of rows, columns and channels in apple image - 512 x 512 x 3 - same as orange image
rows, cols, dim = I1.shape
# convert the generator returned by the pyramid_gaussian function into a tuple of numpy arrays containing the images
# making up individual levels of the pyramid
Apples = tuple(Apple_Gaussian)
# array to hold the apple Gaussian pyramid visualization image showing all the levels in the pyramid
# it contains the same number of rows as the source images but 1.5 times the number of columns to be able to fit the
# original image in the left 512 x 512, the second level of 256 x 256 next to it and all other levels below the latter
composite_image = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)
# Copy the original Apple image which makes up the first layer of the pyramid and (therefore the first element in the
# tuple) into the left 512 x 512 of the visualization image image
composite_image[:rows, :cols, :] = Apples[0]

# copy the remaining levels next to the original image with the biggest 256 x 256 at the top and all the other levels
# below it in decreasing order of size
i_row = 0
for p in Apples[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows
# show the visualization image
fig, ax = plt.subplots()
ax.imshow(composite_image)
plt.show()

# repeat the above Gaussian pyramid visualization process with the orange image
Orange_Gaussian = pyramids.pyramid_gaussian(I2, max_layer=5, downscale=2, multichannel=True)
rows, cols, dim = I2.shape
Oranges = tuple(Orange_Gaussian)
composite_image = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)
composite_image[:rows, :cols, :] = Oranges[0]
i_row = 0
for p in Oranges[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows
fig, ax = plt.subplots()
ax.imshow(composite_image)
plt.show()

# create a Gaussian pyramid for the mask image and repeat the above visualization process
# note that multichannel is set to true here as well since, as mentioned above, the mask is actually a 3 channel array
Mask_Gaussian = pyramids.pyramid_gaussian(M, max_layer=5, downscale=2, multichannel=True)
rows, cols, dim = M.shape
Masks = tuple(Mask_Gaussian)
composite_image = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)
composite_image[:rows, :cols, :] = Masks[0]
i_row = 0
for p in Masks[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows

fig, ax = plt.subplots()
ax.imshow(composite_image)
plt.show()


def laplacian_blend(Apples, Oranges, Masks):
    """

    blend all levels in the apples and oranges Gaussian pyramids using Laplacian blending which produces more natural
    looking results than straightforward averaging by combining information from consecutive levels of the pyramid
    to smoothen intensity changes between the two source images

    source: https://becominghuman.ai/image-blending-using-laplacian-pyramids-2f8e9982077f

    :param tuple(np.ndarray) Apples:
    :param tuple(np.ndarray) Oranges:
    :param tuple(np.ndarray) Masks:
    :rtype tuple(np.ndarray)
    :return: Gaussian pyramid constructed by blending corresponding levels of apples and oranges using the
    corresponding level of masks
    """
    n = len(Apples)
    # blended pyramid levels are stored in a list instead of a tuple since individual levels need to be computed and
    # added one at a time and tuples cannot be modified
    Blended_pyramid = []
    # each iteration processes a couple of consecutive levels in the pyramids so the loop only goes up to the
    # second last level
    for i in range(n - 1):
        # compute the pixel wise difference between consecutive levels of the pyramid by upscaling the smaller level
        # to the same size as the bigger one followed by Gaussian blurring
        # this effectively creates a Laplacian pyramid
        La = Apples[i].astype(np.double) - pyramids.pyramid_expand(Apples[i + 1].astype(np.double), multichannel=True)
        Lo = Oranges[i].astype(np.double) - pyramids.pyramid_expand(Oranges[i + 1].astype(np.double), multichannel=True)

        # combine the apple and orange Laplacian images by a pixel wise weighted average with the weight coming from the
        # corresponding level of the mask pyramid
        L = (1.0 - Masks[i]) * La + Masks[i] * Lo
        # add the blended image as the corresponding level of the blended pyramid
        Blended_pyramid.append(L)

    # blend the last levels of the pyramids without using the Laplacian differencing since there is no more levels to
    # carry that out with
    L = (1.0 - Masks[n - 1]) * Apples[n - 1] + Masks[n - 1] * Oranges[n - 1]
    Blended_pyramid.append(L)
    # convert the list into a tuple
    Blended_pyramid = tuple(Blended_pyramid)
    return Blended_pyramid


def collapse(Blended_pyramid):
    """

    combine all the levels in the blended image pyramid to construct a single blended image
    this process in a sense is the reverse of the Laplacian blending performed in the above function in that
    consecutive levels are added pixel wise starting from the smallest to the largest one

    :param tuple(np.ndarray) Blended_pyramid:
    :return:
    """
    n = len(Blended_pyramid)
    # start with the smallest level, i.e. 16 x 16 x 3
    image = Blended_pyramid[n - 1]
    # iterate from the top of the pyramid to the bottom to go from the smaller to larger images
    # the second argument to np.arange is -1 since the end of iteration is exclusive and we want to go all the way
    # down to 0
    # the third one is also -1 because we want each level to be one less than the previous one
    for i in np.arange(n - 2, -1, -1):
        # upscale the current ( smaller ) image followed by Gaussian blurring and add it pixel wise to the next level
        image = pyramids.pyramid_expand(image, multichannel=True) + Blended_pyramid[i]
    return image


Blended_pyramid = laplacian_blend(Apples, Oranges, Masks)
Blended_image = collapse(Blended_pyramid)

# show the source images as well as the blended one
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(I1)
plt.title("An apple")
plt.subplot(132)
plt.imshow(I2)
plt.title("An orange")
plt.subplot(133)
plt.imshow(Blended_image, cmap="gray")
plt.title("Blending image")
plt.show()
