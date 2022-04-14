import cv2
import numpy as np
import sys

def cleanImage(img):
    kernel = np.ones((5, 5), np.uint8)

    erode = cv2.erode(img, kernel, iterations=2)
    dialate = cv2.dilate(erode, kernel, iterations=2)
    return dialate

def trackImage(img):
    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) != 0:
        for c in cnts:
            if cv2.contourArea(c) > 500:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

def getColorMask(img, color):
    color = color.lower()
    if color == "white":
        lowerBound = np.array([0, 0, 200])
        upperBound = np.array([180, 255, 255])
    elif color == "black":
        lowerBound = np.array([0, 0, 0])
        upperBound = np.array([180, 255, 40])
    elif color == "orange":
        lowerBound = np.array([0, 180, 255])
        upperBound = np.array([20, 255, 255])
    else:
        print("\033[33m" + "INVALID COLOR:", color + '\033[0m')
        sys.exit()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lowerBound, upperBound)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while 1:
    ret, image = cap.read()
    mask = getColorMask(image, "black")

    trackImage(image)

    cv2.imshow("mask", mask)
    cv2.imshow("image", image)

    key = cv2.waitKey(1)

    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break