import cv2
import numpy as np

import matplotlib.pyplot as plt

im_src = plt.imread('./images/paper.jpg')

pts_src = np.array([[94, 167], [291, 170], [40, 364], [310, 384]])
pts_dst = np.array([[0, 0], [300, 0], [0, 400], [300, 400]])

plt.imshow(im_src)
plt.scatter(x=pts_src[:, 0], y=pts_src[:, 1])
plt.show() 

h, status = cv2.findHomography(pts_src, pts_dst)

im_out = cv2.warpPerspective(im_src, h, (im_src.shape[1], im_src.shape[0]))

plt.imshow(im_out)
plt.scatter(x=pts_dst[:, 0], y=pts_dst[:, 1])
plt.show()
plt.imsave('./images/paper_result.jpg', im_out)
