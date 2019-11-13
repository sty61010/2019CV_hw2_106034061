import cv2
import numpy as np
import matplotlib.pyplot as plt

mouse = []
rows = 9
cols = 4
num_points = rows * cols

cnt = 0
# change input image path here
img = cv2.imread('data/data_1.jpg')
height, width, _ = img.shape

def getxy(event, x, y, flags, param):
    global mouse, cnt
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse += [[x, y]]
        np.save('Point2D', np.asarray(mouse)) 
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img, (x, y), 5, (0, 255, 255), -1)
        cv2.imshow('image', img)
        cnt += 1
        if cnt >= num_points:
            cv2.destroyAllWindows()

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', width, height)
cv2.setMouseCallback('image', getxy)

cv2.imshow('image', img)

cv2.waitKey(0)

# change output path here
np.save('data/Point2D_1.txt', np.asarray(mouse))
