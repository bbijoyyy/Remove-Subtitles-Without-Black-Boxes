import cv2 as cv
image_path = 'openCV oneesan.png'
img = cv.imread(image_path)
    



cv.imshow('oneechann', img)
cv.waitKey(0)
cv.destroyAllWindows()