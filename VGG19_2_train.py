# this file is to train a NN for recognize my own face. 

import tensorflow as tf
import numpy as np
import cv2
import os
import random
import sys
from sklearn.model_selection import train_test_split
import time
import VGG_19_2_inference as VGG

original_photo_path = './VGG for photo/Original photo'
modified_photo_path = './VGG for photo/Modified photo'
size = 224

imgs = []
labs = []

def readData(path, h=size, w = size) : 
#    file_list = os.listdir('./')
    for filename in os.listdir(path):
    	if filename.endswith ('.bmp') : 
    		filename = path+'/'+filename
			
    		img = cv2.imread(filename)
			
#			top,bottom,left,right = getPaddingSize(img)
			
			# to increase the image 
			
#			img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0]) 
#			img = cv2.resize(img, (h,w))
			
    		imgs.append(img)
    		labs.append(path)
			
print('start to read orignal photo ......')			
readData(original_photo_path)
print('start to read modified phto  ......')
readData(modified_photo_path)
print('image raed finished')

# convert the image to arrange 
imgs = np.array(imgs)
labs = np.array([[0,1] if lab == original_photo_path else [1,0] for lab in labs])

# randomly allocate traing set and test set. 
train_x, test_x, train_y, test_y = train_test_split(imgs, labs, test_size = 0.05, random_state = random.randint(0,100))

# picture total number, picture  h/w/channel 
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)

# convert to float number 
train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0

print("train_size : %s,  test_size : %s  "%(len(train_x), len(test_x)))

# to setup batch， need to redo to randomly train 
# batch_size = 100
batch_size = 32
num_batch = len(train_x) // batch_size

# set the input / expected output. 
x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

# define the prune probility
keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def cnnTrain() : 
	out = VGG.VGG19_2_inference(x, keep_prob_5)
	
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out.softmax, labels = y_))
	
	train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
	# to compare the label is equal or not, and get average accuracy, tf.cast to convert. 
	
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out.softmax, 1), tf.argmax(y_, 1)), tf.float32))
	
	# save loss and accuracy for tensor board 
	tf.summary.scalar("loss", cross_entropy)
	tf.summary.scalar("accuracy", accuracy)
	merged_summary_op = tf.summary.merge_all() 
	
	# save the data 
	saver = tf.train.Saver() 
	
	with tf.Session() as sess : 
		
		sess.run(tf.global_variables_initializer())
		
		summary_writer = tf.summary.FileWriter('./tmp', graph = tf.get_default_graph())
		train_start_time = time.time()
		
		for n in range(20) :  # iterated 20 times. 
			# get 128 pictures for each batch. 
			for i in range(num_batch) : 
				batch_x = train_x[i*batch_size : (i+1)*batch_size]
				batch_y = train_y[i*batch_size : (i+1)*batch_size]
				
				# start training simultaneiously train 3 avairables, returen 3 
				time_used_for_one_step = time.time()
				_, loss, summary = sess.run([train_step, cross_entropy, merged_summary_op], 
                                        feed_dict = {x:batch_x, y_:batch_y, keep_prob_5 : 0.5})
				summary_writer.add_summary(summary, n*num_batch+i)
				# print the loss 
				time_used_for_one_step = time.time() -  time_used_for_one_step
				print (" loop = %d total batch num = %d  batch (in loop) No. = %d step = %d   loss = %f step time used = % f" %(n, num_batch,i, n*num_batch+i, loss, time_used_for_one_step))
				
				if (n*num_batch+i) % 10 == 0 : 
					# check the accuracy 
					acc = accuracy.eval({x: test_x, y_: test_y, keep_prob_5 : 1.0})
					print (" step = %d   *** accuracy **** = %f "%(n*num_batch+i, acc)) 
					if acc > 0.95 and n > 2 : 
						saver.save (sess, './VGG for photo/train_face.model', global_step = n*num_batch+i)
						print('total time used for training = %f ' %(time.time()- train_start_time))
						sys.exit(0) 
		print('accuracy less than 0.90, exited ! ') 
		
		
cnnTrain()
				
				
	







				

