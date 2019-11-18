import numpy as np
from collections import Counter
import dlib
import cv2
from imutils import face_utils

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

class hair:
    def skin_and_hair_color(self,original_image, rect,gray):
        # try:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)# thresholding light and dark pixels
        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)
        background = cv2.dilate(opening, kernel, iterations=1)# background will be removed
        cv2.imwrite('test.png', background)
        image = original_image
        mask = cv2.imread('test.png')

        mask[mask < 125] = 0
        mask[mask >= 125] = 1
        output = image * (mask == 1)
        # image cropping

        x = rect.left() #x co-ordinate of left corner of rectangle
        y = rect.top() #Y co-ordinate of left corner of rectangle
        w = rect.right() - x # width of rectangle
        h = rect.bottom() - y #height of rectangle
        adj_height = int(0.5 * h)# height is adjusted inorder to take hair inside rectangle
        new_width = int(0.2 * w) # width is adjusted in-order to take less background
        lower_y = y - adj_height # y co-ordinate of rectangle is shifted to nose region to neglect skin
        if lower_y < 0: # if in-case lower_y can out of frame lower_y is rested to 0
            lower_y = 0
        cropped = output[lower_y: lower_y + adj_height, x:x + w - new_width] #cropped image from nose point to the hair.
        cv2.rectangle(original_image,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.rectangle(original_image, (x, lower_y), (w-new_width+x,lower_y + adj_height), (0, 0, 255), 1)
        list_img = [] #list contains RGB values of each pixels of nose region
        pixel_list = [] #list contains added string of RGB values
        for image_x in range(shape[29][1], shape[31][1]):
            for image_y in range(shape[36][0], shape[32][0]):
                z = list(original_image[image_x][image_y])
                list_img.append(z)
        for num in list_img: #string formatting in order to get the most repeated values of RGB
            str0 = num[0]
            real_str0 = "{:03d}".format(str0)
            str1 = num[1]
            real_str1 = "{:03d}".format(str1)
            str2 = num[2]
            real_str2 = "{:03d}".format(str2)
            pixel = (real_str0 + real_str1 + real_str2)
            pixel_list.append(pixel)
        val = Counter(pixel_list).most_common(1) #returns the most repeated value
        value, counts = val[0]
        blue = int(value[0:3]) #open cv returns BGR, extracting values by string formatting
        green = int(value[3:6])
        red = int(value[6:10])
        skin_color = [red, green, blue] #returns RGB of skin color

        tolerence = 100 #tolerence given to skin color to get range of color
        lower_skin_color= np.array([skin_color[0] - tolerence, skin_color[1] - tolerence, skin_color[2] - tolerence])
        upper_skin_color = np.array(
            [skin_color[0] + tolerence, skin_color[1] + tolerence, skin_color[2] + tolerence])  # 154, 180, 234
        mask = cv2.inRange(cropped, lower_skin_color, upper_skin_color) # masking is done on cropped image
        mask_inv = cv2.bitwise_not(mask)
        cropped_image_skin_color_removed = cv2.bitwise_and(cropped, cropped, mask=mask_inv) # return cropped image removing skin color
        cropped_img_pixels = []
        cropped_img_added_pixels = []
        for pixels in cropped_image_skin_color_removed:#taking all pixel values of cropped image
            for x in range(cropped_image_skin_color_removed.shape[1]):
                if pixels[x][0] > 0 and pixels[x][1] > 0 and pixels[x][2] > 0:
                    cropped_img_pixels.append(pixels[x])

        for num in cropped_img_pixels:#string formatting all values of pixels in cropped_img_pixels
            str0 = num[0]
            real_str0 = "{:03d}".format(str0)
            str1 = num[1]
            real_str1 = "{:03d}".format(str1)
            str2 = num[2]
            real_str2 = "{:03d}".format(str2)
            pixel = (real_str0 + real_str1 + real_str2)
            cropped_img_added_pixels.append(pixel)

        val = Counter(cropped_img_added_pixels).most_common(1)#finding most repeated values in cropped_img_added_pixels
        value, counts = val[0]

        blue = int(value[0:3])
        green = int(value[3:6])
        red = int(value[6:10])

        hair_color = [red, green, blue]
        # except Exception as e:
        #     skin_color=None
        #     hair_color=None
        return skin_color, hair_color
