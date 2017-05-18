import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TkAgg')

IMAGE = './corpus/b_dia.jpg'


def main():
    image = cv2.imread(IMAGE)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 20)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    bilateral_blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bilateral_blur, 20, 30)

    _, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
    plot(image, edged)


def plot(*images):
    for i, img in enumerate(images):
        plt.subplot(121 + i)
        plt.imshow(img, cmap='gray')
        plt.xticks([]), plt.yticks([])

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())

    plt.show()


if __name__ == '__main__':
    main()
