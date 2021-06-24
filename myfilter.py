
import numpy as np
import cv2


img = cv2.imread("2020-04-09_13.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 75, 150)
mask = np.zeros((img.shape[0], img.shape[1]))
print("image mask shape", mask.shape)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 5, maxLineGap= 3)
print(lines)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 1)

cv2.imshow("mask", mask)
cv2.imshow("edges", edges )
cv2.imshow("image", gray)
cv2.waitKey(0)

filter = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])



