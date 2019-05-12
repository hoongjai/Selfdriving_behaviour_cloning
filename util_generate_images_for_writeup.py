import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

image = cv2.imread('examples/center_2019_05_11_03_46_52_120_recovery1.jpg')
image = cv2.flip(image,1)
cv2.imwrite('examples/flipped_image.jpg', image)