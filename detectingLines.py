
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import os
import pandas as pd

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
                      joint_kws={'line_kws': {'color': 'magenta'}})
    plt.show()
    slope, intercept, rvalue, pvalue, stderror = linregress(x,y)
    print(f"slope, intercept, rvalue, pvalue, stderror: {slope, intercept, rvalue, pvalue, stderror}")
    return slope, intercept, rvalue, pvalue, stderror

dirpath = r"mix_properties"
ppties = os.listdir(dirpath)
df = []
for ppty in ppties:
    pptypath = os.path.join(dirpath, ppty)
    images = os.listdir(pptypath)
    for i in range(len(images)-1):
        imagepath1 = os.path.join(pptypath, images[i])
        imagepath2 = os.path.join(pptypath, images[i+1])
        image1 = cv2.imread(imagepath1)
        image2 = cv2.imread(imagepath2)
        slope1, intercept1, rvalue1, pvalue1, stderror1 = regrPlot(image1)  ##
        slope2, intercept2, rvalue2, pvalue2, stderror2 = regrPlot(image2)  ##
        df.append([ppty[:5], images[i], images[i+1], round(slope1,5), round(intercept1,5), round(rvalue1,5), round(pvalue1,5), round(stderror1,8),
                   round(slope2,5), round(intercept2,5), round(rvalue2,5), round(pvalue2,5), round(stderror2,8)])

mydf = pd.DataFrame(df, columns= ["property", "image1", "image2", "slope1", "intercept1", "rvalue1", "pvalue1", "stderror1",
                                  "slope2", "intercept2", "rvalue2", "pvalue2", "stderror2"])

# mydf.to_csv("Truepixel_regLines_imgsCompared.csv", index= False)






