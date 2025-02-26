import numpy as np
import cv2

img = np.zeros((512,512,3), dtype= 'uint8')

start_point = (100,100)
end_point = (100,450)

color = (255,250,255)
thickness =9

image = cv2.line(img, start_point, end_point, color, thickness)

cv2.imshow('Drawling Line', image)
cv2.waitKey(0)
cv2.destroyAllWindows()