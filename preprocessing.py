import cv2
import numpy as np

def downscale(images, size):
    """Downscale an array of images and take the mean of the channels."""
    # Convert the images to floating-point numbers
    images = images.astype(np.float32)
    # First, downscale the images to the squared size (changing the aspect ratio)
    square_size = size ** 2
    downscaled_images = np.zeros(shape=(images.shape[0], square_size, square_size))
    for i in range(images.shape[0]):
        downscaled_images[i] = cv2.resize(images[i], (square_size, square_size))
    images = downscaled_images
    # Take the mean over each size * size block of that image to downscale it to size * size
    images = images.reshape((-1, size, size, size, size))
    images = images.transpose((0, 1, 3, 2, 4))
    images = images.reshape((-1, size, size, square_size))
    images = images.mean(-1)
    return images

def discretize(images, bin_count=None, return_bins=False, color_bins=None):
    """Given an array of images, discretize them with a relatively even distribution of discrete values and flatten."""
    size = images.shape[1]
    # If color bins are not provided, calculate them
    if color_bins is None:
        # The color bins should each contain an equal number of values
        bin_percentiles = np.linspace(0, 100, bin_count, endpoint=False)[1:]
        color_bins = np.zeros(shape=(size, size, bin_percentiles.shape[0]))
        for x in range(size):
            for y in range(size):
                # Calculate the percentiles using unique values only, as we want to overrepresent rarely occurring values
                unique_pixel_values = np.unique(images[:, x, y])
                color_bins[x, y] = np.percentile(unique_pixel_values, bin_percentiles)
    # Each of the pixels must be discretized separately, as NumPy does not support separate bins for each pixel
    discrete_images = np.zeros(shape=images.shape, dtype=int)
    for x in range(size):
        for y in range(size):
            discrete_images[:, x, y] = np.digitize(images[:, x, y], color_bins[x, y])
    # Optionally return the color bins along with the discretized images
    if return_bins:
        return discrete_images, color_bins
    else:
        return discrete_images

def encode(discrete_images, bin_count):
    """A function to create a positional scalar encodings for discretized images."""
    # Flatten the images to a 1D vector
    discrete_values = np.reshape(discrete_images, (discrete_images.shape[0], -1))
    encodings = []
    for vec in discrete_values.tolist():
        # Don't use NumPy so that we can have arbitrarily large integers without overflow
        encoding = 0
        for i, value in enumerate(vec):
            encoding += (bin_count ** i) * value
        encodings.append(encoding)
    return encodings

def dense_indices(encodings):
    """Map the very large numbers returned by encode to unique indices."""
    encoding_to_index = {encoding: index for index, encoding in enumerate(set(encodings))}
    return [encoding_to_index[encoding] for encoding in encodings]
