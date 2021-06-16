import cv2
import os
import numpy as np
from natsort import natsorted
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(array1, array2):
    """A cosine value of 0 -> vectors are at 90 degrees to each other (orthogonal) - no match.
     The closer the cosine value to 1, the smaller the angle and the greater the match between vectors."""
    a = array1.reshape(-1)
    b = array2.reshape(-1)
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot/ (norma * normb)
    print(cos)
    return cos


folderpath = "Truepixel/20006_[38.798937210000000_-104.751510800000000]"
imglist = natsorted(os.listdir(folderpath))
n = len(imglist)
# similarity_vector = np.array([])
similarity_vector = []
# print("sim vector initial shape: ", similarity_vector.shape)
for img in imglist:
    print(img)
    imgpath = os.path.join(folderpath, img)
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    print("original shape: ", img.shape)
    rows, cols = img.shape

    sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    print(f"sobel horizontal shape {sobel_horizontal.shape}")
    sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    print(f"sobel vertical shape {sobel_vertical.shape}")
    #
    # cv2.imshow('Original', img)
    # cv2.imshow('Sobel horizontal', sobel_horizontal)
    # cv2.imshow('Sobel vertical', sobel_vertical)
    # cv2.waitKey(0)

    vertical_array  = sobel_vertical.reshape(-1)
    horizontal_array = sobel_horizontal.reshape(-1)
    img_vector = 'sv_'.join([str(img[:-5])])             # dynamic naming.
    img_vector = np.stack((vertical_array, horizontal_array))
    print("img vector shape", img_vector.shape)
    # similarity_vector = np.append([similarity_vector], [img_vector], axis = 0)
    # similarity_vector = np.append([similarity_vector], [img_vector])
    similarity_vector.append(img_vector)
    print("************     **************     *************")

similarity_vector_array = np.array(similarity_vector)
print("similarity vector shape", similarity_vector_array.shape)
# cos_sim = cosine_similarity(similarity_vector_array[0][0].reshape(-1,1), similarity_vector_array[1][0].reshape(-1,1))
# print(cos_sim)
# print(similarity_vector_array[0][0].shape)
# print(similarity_vector_array[1].shape)
# print(similarity_vector_array[2].shape)
# manually compute cosine similarity

print("CALCULATING COSINE SIMILARITY WRT VERTICAL LINES OF IMAGES.")
filter_vertical_scores = []
for i in range(n-1):
    print(f"Images {imglist[i]} and {imglist[i+1]}")
    filter_vertical_scores.append(cosine_sim(similarity_vector_array[i][0], similarity_vector_array[i+1][0]))
    print("***  ++++++++++  ***  ++++++++++  ***")
print("filter_vertical_scores: ", filter_vertical_scores)

print("CALCULATING COSINE SIMILARITY WRT HORIZONTAL LINES OF IMAGES.")
filter_horizontal_scores = []
for i in range(n-1):
    print(f"Images {imglist[i]} and {imglist[i+1]}")
    filter_horizontal_scores.append(cosine_sim(similarity_vector_array[i][1], similarity_vector_array[i+1][1]))
    print("***  ++++++++++  ***  ++++++++++  ***")
print("filter_horizontal_scores: ", filter_horizontal_scores)

################
final_metric = []
for i in range(n-1):
    norm = np.sqrt(((filter_horizontal_scores[i]**2) + (filter_vertical_scores[i] **2))/(n-1))
    final_metric.append(norm)

print("final metric\n", final_metric)

