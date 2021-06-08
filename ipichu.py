

import cv2
# from google.colab.patches import cv2_imshow
import numpy as np

class Preprocessing:
  def inputImage(self, img_path = "pics.jpg" ):
    """Reading the rgb image,
    converting to gray."""
    image = cv2.imread(img_path)
    # height, width = image.shape
    g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return g

    # # let's resize our image to be 150 pixels wide, but in order to
    # # prevent our resized image from being skewed/distorted, we must
    # # first calculate the ratio of the *new* width to the *old* width
    # r = 300 / image.shape[1]
    # dim = (300, int(image.shape[0] * r))
    #
    # image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)

  def cannyEdge(self):
    """Canny edge detection on the ingested image."""
    g = preprocess_obj.inputImage()
    edge = cv2.Canny(g, 30, 30)
    print("Done Canny.")
    return edge

  def inverseForMultiply(self):
    cany = preprocess_obj.cannyEdge()
    inverse = np.where((cany<200),1,0).astype('uint8')  ## type for opencv format.
    inverse = inverse.astype(np.uint8) ## for opencv format.
    # cv2.imshow(inverse)
    print("Inverse done.")
    return inverse

  def bitwiseAnd(self):
    """Bitwise ANDing the images."""
    inv = preprocess_obj.inverseForMultiply()
    result = cv2.bitwise_and(image, image,mask = inv)     ##### change needed here.
    # cv2.imshow(result)
    print("Bitwise and done.")
    return result

  def color_quantization(self, k=8):  # can change the value of k.
    """Quantizing the image."""
    img = preprocess_obj.bitwiseAnd()
    # Transform the image
    data = np.float32(img).reshape((-1, 3))

    # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Implementing K-Means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape(img.shape)
    return res

    # quantized = color_quantization(result, 8)
    # # cv2.imshow(quantized)
    # print("Quantized done.")
  def blurring(self):
    """Applying blurring on quantized image."""
    qntz_img = preprocess_obj.color_quantization()
    blurred = cv2.bilateralFilter(qntz_img, d=7, sigmaColor=200,sigmaSpace=200)
    '''d — Diameter of each pixel neighborhood
    sigmaColor — A larger value of the parameter means larger areas of semi-equal color.
    sigmaSpace –A larger value of the parameter means that farther pixels will 
       influence each other as long as their colors are close enough.
    '''

    return blurred


preprocess_obj = Preprocessing()
cv2.imshow("blurred", preprocess_obj.blurring())
cv2.waitKey(0)
# cv2.imwrite("output.jpg", blurred)

