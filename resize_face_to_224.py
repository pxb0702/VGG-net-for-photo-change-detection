import numpy as np
import cv2
import os
import random
import sys
import time

my_face_path = './TF face recognition 128size/my_face_128'
other_face_path = './TF face recognition 128size/output_dir_other_face_128'

my_face_224_path = './VGG for photo/my_face_224'
other_face_224_path = './VGG for photo/other_face_224'
size = 224

if not os.path.exists(my_face_path): 
    print('the path : ' + my_face_path + " : not exist")
    sys.exit(0)

if not os.path.exists(other_face_path): 
    print('the path : ' + other_face_path + " : not exist")
    sys.exit(0)


if not os.path.exists(my_face_224_path): 
    print('need make (my_face_224_path')
    os.mkdir(my_face_224_path)
else : 
    print('(my_face_224_path already exist')

if not os.path.exists(other_face_224_path): 
    print('need make other_face_224_path')
    os.mkdir(other_face_224_path)
else : 
    print('other_face_224_path already exist')


imgs = []
labs = []

def readData(path, h=size, w = size) : 
#    file_list = os.listdir('./')
    for filename in os.listdir(path):
    	if filename.endswith ('.jpg') : 
    		filename = path+'/'+filename
			
    		img = cv2.imread(filename)
			
#			top,bottom,left,right = getPaddingSize(img)
			
			# to increase the image 
			
#			img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0]) 
#			img = cv2.resize(img, (h,w))
			
    		imgs.append(img)
    		labs.append(path)

def convertFace_224(images, path) : 
    i = 0
    for image in images : 
        temp_image = cv2.resize(image,(size,size))
        file_name = path + '/' + 'image--'+str(i)+ '.bmp'
        cv2.imwrite(file_name,temp_image)
        i = i+1
        if i>3000 :  # only covert 3000 for testing only
            return


print('start to read orignal photo ......')			
readData(my_face_path)
print("start to convert my face - .....")
convertFace_224(imgs, my_face_224_path)
# to clear the image buffer
imgs = []
labs = []
print('start to read modified phto  ......')
readData(other_face_path)
convertFace_224(imgs, other_face_224_path)
print('image convert  finished')
