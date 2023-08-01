import cv2
import numpy as np

# 读入图像
image = cv2.imread('./dataSet/photo_imbd_UTK/61.30_2.jpg', cv2.IMREAD_GRAYSCALE)

# 使用Sobel边缘检测
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

# 计算Sobel结果的模长，再进行开方以便得到清晰的边缘
sobel = np.hypot(sobel_x, sobel_y)
sobel = np.uint8(sobel / sobel.max() * 255) # 归一化

# 使用Canny边缘检测
canny = cv2.Canny(image, 100, 200)

# 展示结果
cv2.imshow('Original', image)
cv2.imshow('Sobel', sobel)
cv2.imshow('Canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
