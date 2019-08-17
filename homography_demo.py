import cv2
import numpy as np

import matplotlib.pyplot as plt

im_src = plt.imread('./images/pepsi.jpg')
cricket_field = plt.imread('./images/cricket_4.jpeg')
# Four corners of the book in source image
pts_src = np.array([[0, 0], [0, 946], [948, 946], [948, 0]])
pts_dst = np.array([[400, 1080], [200, 1350], [840, 1350], [620, 1080]])

plt.imshow(im_src)
plt.scatter(x=pts_src[:, 0], y=pts_src[:, 1])
plt.show() 

h, status = cv2.findHomography(pts_src, pts_dst)

im_out = cv2.warpPerspective(im_src, h, (cricket_field.shape[1], cricket_field.shape[0]))
# print(im_out.shape)
plt.imsave('./images/temp.jpg', im_out)
im_out = plt.imread('./images/temp.jpg')

alpha = 1
beta = 1
dst = cv2.addWeighted(cricket_field, alpha, im_out, beta, 0.0)

plt.imshow(dst)
plt.show()
