import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
IMAGE = './corpus/b_dia.jpg'

def main():
    image = cv2.imread(IMAGE)
    ratio = image.shape[0] / 300.0
    orig = image.copy()
    # image = cv2.resize(image, (400, 500))

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 20)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    bilateral_blur = cv2.bilateralFilter(sharpened, 11, 17, 17)
    edged = cv2.Canny(bilateral_blur, 20, 30)

    # (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    # screenCnt = None

    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edged, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    # cv2.drawContours(image, edged, -1, (0, 255, 0), 3)




    # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]


if __name__ == '__main__':
    main()