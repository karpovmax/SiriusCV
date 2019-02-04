import numpy as np
from skimage.io import imread, imshow, imsave


def colorImage1(input_image, output, classes_file):
    CLASSES = open(classes_file).read().strip().split("\n")

    np.random.seed(123)

    COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3),
                               dtype="uint8")
    COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
    classMap = np.argmax(output, axis=2)
    mask = COLORS[classMap]
    output = ((0.4 * input_image) + (0.6 * mask)).astype("uint8")
    imshow(output)

