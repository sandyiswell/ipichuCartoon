

import cv2
# from google.colab.patches import cv2_imshow
import numpy as np


def color_quantization(img, k= 8):  # can change the value of k.
    """Quantizing the image."""
    # img = preprocess_obj.bitwiseAnd()
    # Transform the image
    data = np.float32(img).reshape((-1, 3))

    # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)

    # Implementing K-Means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 50, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape(img.shape)
    return res


"""Takes an image and does the following.
a - Convert to gray
b - canny edge detection.
c - finding inverse array to multiply.
d - bitwise AND
e - colour quantization and
f - blurring"""

def createCartoon(imgPath):
    image = cv2.imread(imgPath)
    print(image.shape)
    # height, width = image.shape
    g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # canny edge detection.
    edge = cv2.Canny(g, 30, 30)
    print("Done Canny.")
    # cv2_imshow(edge)

    # inverse for multiply.
    inverse = np.where((edge < 200), 1, 0).astype('uint8')  ## type for opencv format.
    inverse = inverse.astype(np.uint8)  ## for opencv format.
    # cv2_imshow(inverse)
    print("Inverse done.")
    result = cv2.bitwise_and(image, image, mask=inverse)  ##### change needed here.
    # cv2.imshow(result)
    # cv2_imshow(result)
    print("Bitwise and done.")

    reslt = color_quantization(result)
    # cv2_imshow(reslt)
    print("Quantization done.")

    # blurring.
    blurred = cv2.bilateralFilter(reslt, d=5, sigmaColor=10, sigmaSpace=10)
    # cv2_imshow(blurred)
    print("Blurring done.")
    # cv2.imshow("blurred", blurred)
    # cv2.waitKey(0)

    #########################################
    # creating the filter.
    # filter is 3x3.
    # creating the filter.
    filter = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]]) # Box Filter

    # splitting channels.
    b, g, r = cv2.split(blurred)
    print("shape of blue channel", b.shape)
    print("shape of green channel", g.shape)
    print("shape of red channel", r.shape)


    channels = [b, g, r]
    ch = 0

    for channel in channels:
        for h in range(channel.shape[0]-2):
            for w in range(channel.shape[1]-2):
                # maximum pixel value which can be accommodated.
                max_pix_val = 150
                if channel[h +1, w +1] < max_pix_val:
                    instance = channel[h:h+3, w:w+3]
                    avg = int(np.sum(instance * filter)/ 8)
                    channel[h +1, w +1] = avg
        ch = ch +1
        print(f"channel {ch} done.")

    merged = cv2.merge([b, g, r])

    cv2.imshow("merged", merged )
    # cv2.imwrite("devsena.jpg", merged)
    cv2.waitKey(0)

# Run the program by loading image.
createCartoon('input/actress.jpg')

