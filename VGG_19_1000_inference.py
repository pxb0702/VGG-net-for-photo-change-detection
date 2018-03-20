from datetime import datetime
import math
import time
import tensorflow as tf 


class VGG19_19_100_inference() : 
    def __init__(self, input_op): 
        self.p = []
        
        # 1st convolution layer -  2 convolution + 1 max pool 
        self.conv1_1 = conv_op(input_op, name="conv1_1", kh = 3, kw = 3, n_out = 64, dh=1, dw = 1, p=self.p )
        self.conv1_2 = conv_op(self.conv1_1, name="conv1_2", kh = 3, kw = 3, n_out = 64, dh=1, dw = 1, p=self.p )
        self.pool1 = mpool_op(self.conv1_2, name = "pool1", kh = 2, kw = 2, dh = 2, dw = 2)

        # 2nd convolution layer  - 2 convolution + 1 max pool,  
        self.conv2_1 = conv_op(self.pool1, name="conv2_1", kh = 3, kw = 3, n_out = 128, dh=1, dw = 1, p=self.p )
        self.conv2_2 = conv_op(self.conv2_1, name="conv2_2", kh = 3, kw = 3, n_out = 128, dh=1, dw = 1, p=self.p )
        self.pool2 = mpool_op(self.conv2_2, name = "pool2", kh = 2, kw = 2, dh = 2, dw = 2)

        # 3rd convolution layer - 4 convolution layer + 1 max pooling 
        self.conv3_1 = conv_op(self.pool2, name="conv3_1", kh = 3, kw = 3, n_out = 256, dh=1, dw = 1, p=self.p )
        self.conv3_2 = conv_op(self.conv3_1, name="conv3_2", kh = 3, kw = 3, n_out = 256, dh=1, dw = 1, p=self.p )
        self.conv3_3 = conv_op(self.conv3_2, name="conv3_3", kh = 3, kw = 3, n_out = 256, dh=1, dw = 1, p=self.p )
        self.conv3_4 = conv_op(self.conv3_3, name="conv3_4", kh = 3, kw = 3, n_out = 256, dh=1, dw = 1, p=self.p )
        self.pool3 = mpool_op(self.conv3_4, name = "pool3", kh = 2, kw = 2, dh = 2, dw = 2)
        
        # 4th convolution layer - 4 convolution layer + 1 max pooling 
        self.conv4_1 = conv_op(self.pool3, name="conv4_1", kh = 3, kw = 3, n_out = 512, dh=1, dw = 1, p=self.pp )
        self.conv4_2 = conv_op(self.conv4_1, name="conv4_2", kh = 3, kw = 3, n_out = 512, dh=1, dw = 1, p=self.pp )
        self.conv4_3 = conv_op(self.conv4_2, name="conv4_3", kh = 3, kw = 3, n_out = 512, dh=1, dw = 1, p=self.pp )
        self.conv4_4 = conv_op(self.conv4_3, name="conv4_4", kh = 3, kw = 3, n_out = 512, dh=1, dw = 1, p=self.pp )
        self.pool4 = mpool_op(self.conv4_4, name = "pool4", kh = 2, kw = 2, dh = 2, dw = 2) 

        # 5th convolution layer - 4 convolution layer + 1 max pooling 
        self.conv5_1 = conv_op(self.pool4, name="conv5_1", kh = 3, kw = 3, n_out = 512, dh=1, dw = 1, p=self.pp )
        self.conv5_2 = conv_op(self.conv5_1, name="conv5_2", kh = 3, kw = 3, n_out = 512, dh=1, dw = 1, p=self.pp )
        self.conv5_3 = conv_op(self.conv5_2, name="conv5_3", kh = 3, kw = 3, n_out = 512, dh=1, dw = 1, p=self.pp )
        self.conv5_4 = conv_op(self.conv5_3, name="conv5_4", kh = 3, kw = 3, n_out = 512, dh=1, dw = 1, p=self.pp )
        self.pool5 = mpool_op(self.conv5_4, name = "pool5", kh = 2, kw = 2, dh = 2, dw = 2) 

        # reshape for Full connectin network
        shp = self.pool5.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        resh1 = tf.reshape(pool5,[-1, flattened_shape], name = "resh1")

        # FC 1
        self.fc6 = fc_op(resh1, name = "fc6", n_out = 4096, p = self.pp)
        self.fc6_drop = tf.nn.dropout(self.fc6, keep_prob, name= "fc6_drop")
        #FC 2
        self.fc7 = fc_op (self.fc6_drop, name = "fc7", n_out=4096, p = self.pp)
        self.fc7_drop = tf.nn.dropout(self.fc7, keep_prob, name = "fc7_drop")
        #FC 3
        self.fc8 = fc_op (self.fc7_drop, name = "fc8", n_out = 1000, p=self.pp)
        # Full result output
        self.softmax = tf.nn.softmax(self.fc8)
        # predictions
        self.predictions = tf.argmax(self.softmax, 1)

        
    def conv_op(self, input_op, name, kh, kw, n_out, dh, dw, p) : 
        n_in = input_op.get_shape()[-1].value

        with tf.name_scope(name) as scope : 
            kernel = tf.get_variable(scope+"w", 
                shape = [kh, kw, n_in,n_out], 
                dtype = tf.float32, initializer= tf.contrib.layers.xavier_initializer_conv2d()) 
            
            conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding = "SAME")
            bias_init_val = tf.constant(0.0, shape = [n_out], dtype = tf.float32)
            biases = tf.Variable(bias_init_val, trainable=True, name = "b")

            z = tf.nn.bias_add(conv, biases)
            activation = tf.nn.relu(z, name = scope)

            p += [kernel, biases]

            return activation


    def fc_op (self, input_op, name, n_out, p) : 
        n_in = input_op.get_shape()[-1].value

        with tf.name_scope(name) as scope : 
            kernel = tf.get_variable(scope+"w", 
                shape = [n_in,n_out], 
                dtype = tf.float32, initializer= tf.contrib.layers.xavier_initializer()) 
            
            biases = tf.Variable(tf.constant(0.1, shape = [n_out], dtype = tf.float32), trainable=True, name = "b")

            activation = tf.nn.relu_layer(input_op, kernel, biases, name = scope)

            p += [kernel, biases]

            return activation    

    def mpool_op(self,input_op, name, kh, kw, dh, dw) : 
        return tf.nn.max_pool(input_op, 
                        ksize = [1,kh,kw,1], 
                        strides = [1,dh,dw,1], 
                        padding = 'SAME', 
                        name = name)


def time_tensorflow_run (session, target, feed, info_string) : 
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range (num_batches + num_steps_burn_in) : 
        start_time = time.time()
        _ = session.run(target, feed_dict = feed)
        duration = time.time() - start_time 
        print('%d sub-step : %.3f seconds used '%(i, duration))
        if i >= num_steps_burn_in : 
            if not i%10 : 
                print ("%s, step %d, duration = %.3f" % 
                    (datetime.now(), i-num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration

    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)

    print('%s :  %s  across  %d  steps,  %.3f +/- %.3f  sec / batch ' %
           (datetime.now(), info_string, num_batches, mn, sd))

def run_benchmark() : 
    with tf.Graph().as_default() : 
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,
                                             image_size,
                                             image_size,
                                             3] , 
                                             dtype=tf.float32, 
                                             stddev = 1e-1))

        keep_prob = tf.placeholder(tf.float32) 
        predictions, softmax, fc8, p = inference_op(images, keep_prob)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        print("forward evaluation started \n")
        time_tensorflow_run(sess, predictions, {keep_prob : 1.0}, "Foward")    

        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, p)
        print("backward evaluation started \n")
        time_tensorflow_run (sess, grad, {keep_prob : 0.5}, "Foward - backward")

run_benchmark()


