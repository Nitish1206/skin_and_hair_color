import numpy as np
import cv2
from collections import Counter
import dlib
import cv2
from imutils import face_utils
from get_color import hair

final_color = hair()
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
# cap = cv2.VideoCapture(0)
# while(True):
#     ret, original_image = cap.read()
#     gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
#     rects = detector(gray, 0)
#     if len(rects) > 0:
#         for (i, rect) in enumerate(rects):
#             color = final_color.skin_and_hair_color(original_image, rect, gray)
#
#             print('skin color', color[0], i)
#             print('hair color', color[1], i)
#             cv2.rectangle(original_image,(530,50),(630,150),(color[1][2],color[1][1],color[1][0]),-1)
#             cv2.rectangle(original_image, (400, 50), (500, 150), (color[0][2], color[0][1], color[0][0]), -1)
#
#     cv2.imshow("output", original_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


original_image = cv2.imread("image path")
gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 0)
if len(rects) > 0:
    for (i, rect) in enumerate(rects):
        color = final_color.skin_and_hair_color(original_image, rect, gray)

        print('skin color', color[0], i)
        print('hair color', color[1], i)
        cv2.rectangle(original_image,(530,50),(630,150),(color[1][2],color[1][1],color[1][0]),-1)
        cv2.rectangle(original_image, (400, 50), (500, 150), (color[0][2], color[0][1], color[0][0]), -1)
cv2.imshow("output", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
