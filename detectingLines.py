
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import os
import pandas as pd



# img1 = cv2.imread(r"Truepixel\20018_[38.725973600000000_-104.667953000000000]\2019-03-27_7.jpeg")
# img2 = cv2.imread(r"Truepixel\20018_[38.725973600000000_-104.667953000000000]\2019-04-27_8.jpeg")
# img3 = cv2.imread(r"Truepixel\20018_[38.725973600000000_-104.667953000000000]\2019-08-27_12.jpeg")

def createLines(img):
    """Takes an image, finds Canny output and finds lines up on it."""
    mask = np.zeros((img.shape[0], img.shape[1]))
    print("image mask shape", mask.shape)
    edges = cv2.Canny(img, 75, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 25, maxLineGap=3)
    print(lines)
    print(f"Length of lines: {len(lines)}")
    xpoints = []
    ypoints = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        xpoints.append(int(x1))
        xpoints.append(int(x2))
        ypoints.append(int(y1))
        ypoints.append(int(y2))
        # cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return xpoints, ypoints

def regrPlot(img):
    """Create regression plots based on input line coordinates.
    Returns slope, intercept, rvalue, pvalue, stderror of the regression line."""
    x, y = createLines(img)
    # m, b = np.polyfit(x, y, 1)
    # sns.set(style="white", color_codes=True)
    # sns.regplot(x, y)
    # print("m : ", m)
    # plt.plot(x, m * x + b)
    g = sns.jointplot(x, y,  kind='reg',
                      joint_kws={'line_kws': {'color': 'yellow'}})
    plt.show()
    slope, intercept, rvalue, pvalue, stderror = linregress(x,y)
    print(f"slope, intercept, rvalue, pvalue, stderror: {slope, intercept, rvalue, pvalue, stderror}")
    return slope, intercept, rvalue, pvalue, stderror

# regrPlot(img1)
# regrPlot(img2)
# regrPlot(img3)

dirpath = r"single_property"
ppties = os.listdir(dirpath)
df = []
for ppty in ppties:
    pptypath = os.path.join(dirpath, ppty)
    images = os.listdir(pptypath)
    for image in images:
        imagepath = os.path.join(pptypath, image)
        image1 = cv2.imread(imagepath)
        slope, intercept, rvalue, pvalue, stderror = regrPlot(image1)  ##
        df.append([ppty[:5], image, round(slope,5), round(intercept,5), round(rvalue,5), round(pvalue,5), round(stderror,8)])

mydf = pd.DataFrame(df, columns= ["property", "image", "slope", "intercept", "rvalue", "pvalue", "stderror"])
# mydf.to_csv("Truepixel_regLines.csv", index= False)






