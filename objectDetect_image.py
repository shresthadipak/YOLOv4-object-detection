import cv2
from objectDetectionModule import objectDetector

detector = objectDetector()

img = cv2.imread('images/image1.jpg')

h, w, _ = img.shape

# define screen resolution
screen_res = 1280, 720
scale_width = screen_res[0] / img.shape[1]
scale_height = screen_res[1] / img.shape[0]
scale = min(scale_width, scale_height)

img = cv2.resize(img, (int(w/2 * scale), int(h/2 * scale)), cv2.INTER_AREA)

detected = detector.object_detect(img)

cv2.imshow("Yolov4", detected)

cv2.waitKey(0)
