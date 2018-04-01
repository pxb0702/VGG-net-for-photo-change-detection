# this file is to train a NN for recognize my own face. 

import tensorflow as tf
import numpy as np
import cv2
import os
import random
import sys
from sklearn.model_selection import train_test_split
import time

my_face_path = './my_face_128'
other_face_path = './output_dir_other_face_128'
size = 128

imgs = []
labs = []

def getPaddingSize (img) : 
	h,w,_ = img.shape
	top, bottom, left, right = (0,0,0,0)
	longest = max(h,w)
	
	if w<longest : 
		tmp = longest - w
		# // means ingerated division 
		left = tmp//2
		right = tmp - left
	elif h < longest : 
		tmp = longest - h
		top = tmp//2
		bottom = tmp - top
	else : 
		pass
	return top, bottom, left, right
	
def readData(path, h=size, w = size) : 
	for filename in os.listdir(path) : 
		if filename.endswith ('.jpg') : 
			filename = path+'/'+filename
			
			img = cv2.imread(filename)
			
			top,bottom,left,right = getPaddingSize(img)
			
			# to increase the image 
			
			img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0]) 
			img = cv2.resize(img, (h,w))
			
			imgs.append(img)
			labs.append(path)
			
print('start to read my face ......')			
readData(my_face_path)
print('start to read other face ......')
readData(other_face_path)
print('image raed finished')

# convert the image to arrange 
imgs = np.array(imgs)
labs = np.array([[0,1] if lab == my_face_path else [1,0] for lab in labs])

# randomly allocate traing set and test set. 
train_x, test_x, train_y, test_y = train_test_split(imgs, labs, test_size = 0.05, random_state = random.randint(0,100))

# picture total number, picture  h/w/channel 
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)

# convert to float number 
train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0

print("train_size : %s,  test_size : %s  "%(len(train_x), len(test_x)))

# to setup batch 
batch_size = 100
num_batch = len(train_x) // batch_size

# set the input / expected output. 
x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

# define the prune probility
keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

# to randomize the weight 
def weightVariable(shape) : 
	init = tf.random_normal(shape, stddev = 0.01)
	return tf.Variable(init)

def biasVariable(shape) : 
	init = tf.random_normal(shape)
	return tf.Variable(init)

def conv2d(x, W) : 
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = "SAME")
	
def maxPool(x) : 
	return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")
	
def dropout(x, keep) : 
	return tf.nn.dropout(x, keep)
	
def cnnLayer() : 
	
	# 1st layer 
	W1 = weightVariable([3,3,3,32]) # core : 3x3, channel = 3, core number = 32 
	b1 = biasVariable([32])
	# convolution ? where come from the x ????
	conv1 = tf.nn.relu(conv2d(x, W1) + b1)
	# pooling : 
	pool1 = maxPool(conv1)
	# reduce the redundent, randomly select some parameters not updated. 
	drop1 = dropout(pool1, keep_prob_5)

	# 2nd layer 
	W2 = weightVariable([3,3,32,64]) # core : 3x3, channel = 3, core number = 32 
	b2 = biasVariable([64])
	# convolution ? where come from the x ????
	conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
	# pooling : 
	pool2 = maxPool(conv2)
	# reduce the redundent, randomly select some parameters not updated. 
	drop2 = dropout(pool2, keep_prob_5)


	# 3rd layer 
	W3 = weightVariable([3,3,64,64]) # core : 3x3, channel = 3, core number = 32 
	b3 = biasVariable([64])
	# convolution ? where come from the x ????
	conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
	# pooling : 
	pool3 = maxPool(conv3)
	# reduce the redundent, randomly select some parameters not updated. 
	drop3 = dropout(pool3, keep_prob_5)

	# full connection layer - 1st layer. 
	Wf_1 = weightVariable([16*16*64, 512])
	bf_1 = biasVariable([512])
	drop3_flat_1 = tf.reshape(drop3, [-1, 16*16*64])
	dense_1 = tf.nn.relu(tf.matmul(drop3_flat_1, Wf_1) + bf_1)
	dropf_1 = dropout(dense_1, keep_prob_75)
	
# full connection layer - 2nd layer. 
	Wf_2 = weightVariable([16*16*64, 512])
	bf_2 = biasVariable([512])
	drop3_flat_2 = tf.reshape(drop3_flat_1, [-1, 16*16*64])
	dense_2 = tf.nn.relu(tf.matmul(drop3_flat_2, Wf_2) + bf_2)
	dropf_2 = dropout(dense_2, keep_prob_75)


	# output layer 
	Wout = weightVariable([512, 2])
	bout = weightVariable([2])
	# out = tf.matmul(dropf, Wout) + bout 
	out = tf.add(tf.matmul(dropf_2, Wout), bout)
	return out
	
def cnnTrain() : 
	out = cnnLayer()
	
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out, labels = y_))
	
	train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
	# to compare the label is equal or not, and get average accuracy, tf.cast to convert. 
	
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
	
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
		
		for n in range(20) :  # iterated 10 times. 
			# get 128 pictures for each batch. 
			for i in range(num_batch) : 
				batch_x = train_x[i*batch_size : (i+1)*batch_size]
				batch_y = train_y[i*batch_size : (i+1)*batch_size]
				
				# start training simultaneiously train 3 avairables, returen 3 
				time_used_for_one_step = time.time()
				_, loss, summary = sess.run([train_step, cross_entropy, merged_summary_op], feed_dict = {x:batch_x, y_:batch_y, keep_prob_5 : 0.5, keep_prob_75: 0.75})
				summary_writer.add_summary(summary, n*num_batch+i)
				# print the loss 
				time_used_for_one_step = time.time() -  time_used_for_one_step
				print (" loop = %d total batch num = %d  batch (in loop) No. = %d step = %d   loss = %f step time used = % f" %(n, num_batch,i, n*num_batch+i, loss, time_used_for_one_step))
				
				if (n*num_batch+i) % 100 == 0 : 
					# check the accuracy 
					acc = accuracy.eval({x: test_x, y_: test_y, keep_prob_5 : 1.0, keep_prob_75 : 1.0})
					print (" step = %d   *** accuracy **** = %f "%(n*num_batch+i, acc)) 
					if acc > 0.98 and n > 2 : 
						saver.save (sess, './train_face.model', global_step = n*num_batch+i)
						print('total time used for training = %f ' %(time.time()- train_start_time))
						sys.exit(0) 
		print('accuracy less than 0.98, exited ! ') 
		
		
cnnTrain()
				
				
	







				

