from datetime import datetime
import math
import time
import tensorflow as tf 
import VGG_19_2_inference as VGG

batch_size = 32
num_batches = 100

# this is to test the 2-output VGG 19 inference

def time_tensorflow_run (session, target,info_string) : 
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range (num_batches + num_steps_burn_in) : 
        start_time = time.time()
        _ = session.run(target)
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

        #keep_prob = tf.placeholder(tf.float32) 
        # predictions, softmax, fc8, p = inference_op(images, keep_prob)
        keep_prob = 1.0
        VGG_19_net = VGG.VGG19_2_inference(images, keep_prob)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        print("forward evaluation started \n")
        time_tensorflow_run(sess, VGG_19_net.predictions, "Foward")    

        objective = tf.nn.l2_loss(VGG_19_net.fc8)
        grad = tf.gradients(objective, VGG_19_net.p)
        print("backward evaluation started \n")
        # time_tensorflow_run (sess, grad, {keep_prob : 0.5}, "Foward - backward")
        time_tensorflow_run (sess, grad, "Foward - backward")

run_benchmark()


