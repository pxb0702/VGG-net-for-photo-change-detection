#  this file is to generate .BMP file for simulating the training input. 

import cv2
import dlib
import os
import sys
import random
import datetime
import tensorflow as tf 
import numpy as np

image_size = 224


# the directories where the training / test image locate. 
original_photo_path = './VGG for photo/Original photo'
modified_photo_path = './VGG for photo/Modified photo'

# check the directory, if not exist, make the directories
if not os.path.exists(original_photo_path): 
    print('need make original photo path')
    os.mkdir(original_photo_path)
else : 
    print('orignal photo path already exist')


if not os.path.exists(modified_photo_path): 
    print('need make modified photo path')
    os.mkdir(modified_photo_path)
else : 
    print('modified photo path already exist')


# define how many photos to be generated
num_original_photo = 500
num_modified_photo = 500

# to get date and time
today_date = datetime.date.today()
datestr = str(today_date)

# to generate  photo file in original

for i in range(num_original_photo): 
    img_r = np.random.randint(0,256,size=(image_size,image_size))
    img_g = np.random.randint(0,256,size=(image_size,image_size))
    img_b = np.random.randint(0,256,size=(image_size,image_size))
 #   img = tf.concat(axis=2, values=[img_r,img_g,img_b])
    
    img = np.dstack((img_r, img_g, img_b))
#    assert img.get_shape().as_list()[1:] == [224, 224, 3]
    file_name = original_photo_path + '/' + datestr + "--" + str(i)+'.bmp'
    cv2.imwrite(file_name, img)

# to generate the file in modified 
for i in range(num_modified_photo): 
    img_r = np.random.randint(0,256,size=(image_size,image_size))
    img_g = np.random.randint(0,256,size=(image_size,image_size))
    img_b = np.random.randint(0,256,size=(image_size,image_size))
 #   img = tf.concat(axis=2, values=[img_r,img_g,img_b])
    
    img = np.dstack((img_r, img_g, img_b))
#    assert img.get_shape().as_list()[1:] == [224, 224, 3]
    file_name = modified_photo_path + '/' + datestr + "--" + str(i)+'.bmp'
    cv2.imwrite(file_name, img)

	