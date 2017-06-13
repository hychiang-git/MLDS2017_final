import numpy as np
import tensorflow as tf, tqdm
from tensorflow.examples.tutorials.mnist import input_data


class MNIST_EX(object):
    def __init__(self, x, y_):
        # Small epsilon value for the BN transform
        epsilon = 1e-3

        # Generate predetermined random weights so the networks are similarly initialized
        w1_initial = np.random.normal(size=(784,100)).astype(np.float32)
        w2_initial = np.random.normal(size=(100,100)).astype(np.float32)
        w3_initial = np.random.normal(size=(100,10)).astype(np.float32)
        
        # Layer 1 without BN
        self.w1 = tf.Variable(w1_initial)
        self.b1 = tf.Variable(tf.zeros([100]))
        self.z1_BN = tf.matmul(x, self.w1)
        # whitening
        self.batch_mean1, self.batch_var1 = tf.nn.moments(self.z1_BN, [0])
        self.z1_hat = self.z1_BN - self.batch_mean1
        #self.z1_cov = tf.matmul(tf.transpose(self.z1_hat), self.z1_hat) / tf.cast(tf.shape(x)[1], tf.float32)
        self.z1_cov = tf.matmul(tf.transpose(self.z1_hat), self.z1_hat) / 60
        print "----------------------------"
        print self.z1_cov.get_shape().as_list() 
        print tf.shape(x)
        print "----------------------------"
        self.z1_conv_inv = tf.matrix_inverse(self.z1_cov)
        self.W_white = tf.cholesky(self.z1_conv_inv)
        self.z1_white = tf.matmul(self.z1_BN, self.W_white)
        #self.z1_white = tf.matmul(self.z1_hat, tf.sqrt(self.z1_conv_inv))
        #S, U, V = tf.svd(self.z1_cov)
        #self.z1_rot = tf.matmul(self.z1_cov, U)
        #self.z1_white = self.z1_rot / tf.sqrt(S + epsilon)
        self.z1 = self.z1_white + self.b1
        self.l1 = tf.nn.sigmoid(self.z1)

        # Layer 2 without BN
        self.w2 = tf.Variable(w2_initial)
        self.b2 = tf.Variable(tf.zeros([100]))
        self.z2 = tf.matmul(self.l1, self.w2) + self.b2
        self.l2 = tf.nn.sigmoid(self.z2)

        # Softmax
        self.w3 = tf.Variable(w3_initial)
        self.b3 = tf.Variable(tf.zeros([10]))
        self.y  = tf.nn.softmax(tf.matmul(self.l2, self.w3) + self.b3)
        print "----------------------------"
        print self.y.get_shape().as_list()
        print y_.get_shape().as_list()
        print tf.shape(x)
        print "----------------------------"
        # Loss, optimizer and predictions
        self.cross_entropy = -tf.reduce_sum(y_*tf.log(self.y))

        # Optimizer
        self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy)

        # Accuracy
        self.correct_prediction = tf.equal(tf.arg_max(self.y,1),tf.arg_max(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32))





class MNIST(object):
    def __init__(self, x, y_):
        
        # Generate predetermined random weights so the networks are similarly initialized
        w1_initial = np.random.normal(size=(784,100)).astype(np.float32)
        w2_initial = np.random.normal(size=(100,100)).astype(np.float32)
        w3_initial = np.random.normal(size=(100,10)).astype(np.float32)

        # Layer 1 without BN
        self.w1 = tf.Variable(w1_initial)
        self.b1 = tf.Variable(tf.zeros([100]))
        self.z1 = tf.matmul(x, self.w1) + self.b1
        self.l1 = tf.nn.sigmoid(self.z1)

        # Layer 2 without BN
        self.w2 = tf.Variable(w2_initial)
        self.b2 = tf.Variable(tf.zeros([100]))
        self.z2 = tf.matmul(self.l1, self.w2) + self.b2
        self.l2 = tf.nn.sigmoid(self.z2)

        # Softmax
        self.w3 = tf.Variable(w3_initial)
        self.b3 = tf.Variable(tf.zeros([10]))
        self.y  = tf.nn.softmax(tf.matmul(self.l2, self.w3) + self.b3)

        # Loss, optimizer and predictions
        self.cross_entropy = -tf.reduce_sum(y_*tf.log(self.y))

        # Optimizer
        self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy)

        # Accuracy
        self.correct_prediction = tf.equal(tf.arg_max(self.y,1),tf.arg_max(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32))




class MNIST_BN(object):
    def __init__(self, x, y_):
        # Small epsilon value for the BN transform
        epsilon = 1e-3


        # Generate predetermined random weights so the networks are similarly initialized
        w1_initial = np.random.normal(size=(784,100)).astype(np.float32)
        w2_initial = np.random.normal(size=(100,100)).astype(np.float32)
        w3_initial = np.random.normal(size=(100,10)).astype(np.float32)

        # Layer 1 with BN
        self.w1_BN = tf.Variable(w1_initial)
        self.z1_BN = tf.matmul(x, self.w1_BN)

        self.scale1 = tf.Variable(tf.ones([100]))
        self.beta1 = tf.Variable(tf.zeros([100]))
        self.batch_mean1, self.batch_var1 = tf.nn.moments(self.z1_BN, [0])
        self.z1_hat = (self.z1_BN - self.batch_mean1) / tf.sqrt(self.batch_var1 + epsilon)
        self.BN1 = self.scale1 * self.z1_hat + self.beta1
        self.l1_BN = tf.nn.sigmoid(self.BN1)


        # Layer 2 with BN, using Tensorflows built-in BN function
        self.w2_BN = tf.Variable(w2_initial)
        self.z2_BN = tf.matmul(self.l1_BN, self.w2_BN)
        
        self.scale2 = tf.Variable(tf.ones([100]))
        self.beta2 = tf.Variable(tf.zeros([100]))
        self.batch_mean2, self.batch_var2 = tf.nn.moments(self.z2_BN, [0])
        self.z2_hat = (self.z2_BN - self.batch_mean2) / tf.sqrt(self.batch_var2 + epsilon)
        self.BN2 = self.scale2 * self.z2_hat + self.beta2
        #self.BN2 = tf.nn.batch_normalization(self.z2_BN, self.batch_mean2, self.batch_var2, self.beta2, self.scale2, epsilon)
        self.l2_BN = tf.nn.sigmoid(self.BN2)

        # Softmax
        self.w3_BN = tf.Variable(w3_initial)
        self.b3_BN = tf.Variable(tf.zeros([10]))
        self.y_BN  = tf.nn.softmax(tf.matmul(self.l2_BN, self.w3_BN) + self.b3_BN)

        # Loss, optimizer and predictions
        self.cross_entropy_BN = -tf.reduce_sum(y_*tf.log(self.y_BN))
        
        #Optimizer
        self.train_step_BN = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy_BN)

        # Accuracy
        self.correct_prediction_BN = tf.equal(tf.arg_max(self.y_BN,1), tf.arg_max(y_,1))
        self.accuracy_BN = tf.reduce_mean(tf.cast(self.correct_prediction_BN, tf.float32))





class MNIST_BN_MVAVG(object):

    def __init__(self, x, y_, is_training):

        # Generate predetermined random weights so the networks are similarly initialized
        w1_initial = np.random.normal(size=(784,100)).astype(np.float32)
        w2_initial = np.random.normal(size=(100,100)).astype(np.float32)
        w3_initial = np.random.normal(size=(100,10)).astype(np.float32)

        # Layer 1
        self.w1_BAV = tf.Variable(w1_initial)
        self.z1_BAV = tf.matmul(x, self.w1_BAV)
        self.BAV1= self.batch_norm_wrapper(self.z1_BAV, is_training)
        self.l1_BAV = tf.nn.sigmoid(self.BAV1)

        #Layer 2
        self.w2_BAV = tf.Variable(w2_initial)
        self.z2_BAV = tf.matmul(self.l1_BAV, self.w2_BAV)
        self.BAV2= self.batch_norm_wrapper(self.z2_BAV, is_training)
        self.l2_BAV = tf.nn.sigmoid(self.BAV2)

        # Softmax
        self.w3_BAV = tf.Variable(w3_initial)
        self.b3_BAV = tf.Variable(tf.zeros([10]))
        self.y_BAV  = tf.nn.softmax(tf.matmul(self.l2_BAV, self.w3_BAV) + self.b3_BAV)

        # Loss, Optimizer and Predictions
        self.cross_entropy_BAV = -tf.reduce_sum(y_*tf.log(self.y_BAV))

        self.train_step_BAV = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy_BAV)

        self.correct_prediction_BAV = tf.equal(tf.arg_max(self.y_BAV,1),tf.arg_max(y_,1))
        self.accuracy_BAV = tf.reduce_mean(tf.cast(self.correct_prediction_BAV, tf.float32))




    # this is a simpler version of Tensorflow's 'official' version. See:
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L102
    def batch_norm_wrapper(self, inputs, is_training, decay = 0.99):
        # Small epsilon value for the BN transform
        epsilon = 1e-3
    
        self.scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        self.beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        self.pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        self.pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    
        def training():
            self.batch_mean, self.batch_var = tf.nn.moments(inputs,[0])
            self.train_mean = tf.assign(self.pop_mean,
                                   self.pop_mean * decay + self.batch_mean * (1 - decay))
            self.train_var = tf.assign(self.pop_var,
                                   self.pop_var  * decay + self.batch_var  * (1 - decay))

            with tf.control_dependencies([self.train_mean, self.train_var]):
                return tf.nn.batch_normalization(inputs,
                    self.batch_mean, self.batch_var, self.beta, self.scale, epsilon)
        def testing():
            return tf.nn.batch_normalization(inputs,
                self.pop_mean, self.pop_var, self.beta, self.scale, epsilon)

        return tf.cond(is_training, training, testing)
    

class MNIST_BRN(object):

    def __init__(self, x, y_, is_training):
        self.steps = tf.Variable(0, name='global_step', trainable=False)

        # Generate predetermined random weights so the networks are similarly initialized
        w1_initial = np.random.normal(size=(784,100)).astype(np.float32)
        w2_initial = np.random.normal(size=(100,100)).astype(np.float32)
        w3_initial = np.random.normal(size=(100,10)).astype(np.float32)

        # Layer 1
        self.w1_BRN = tf.Variable(w1_initial)
        self.z1_BRN = tf.matmul(x, self.w1_BRN)
        self.BRN1 = self.batch_renorm_wrapper(self.z1_BRN, is_training)
        self.l1_BRN = tf.nn.sigmoid(self.BRN1)

        #Layer 2
        self.w2_BRN = tf.Variable(w2_initial)
        self.z2_BRN = tf.matmul(self.l1_BRN, self.w2_BRN)
        self.BRN2 = self.batch_renorm_wrapper(self.z2_BRN, is_training)
        self.l2_BRN = tf.nn.sigmoid(self.BRN2)

        # Softmax
        self.w3_BRN = tf.Variable(w3_initial)
        self.b3_BRN = tf.Variable(tf.zeros([10]))
        self.y_BRN  = tf.nn.softmax(tf.matmul(self.l2_BRN, self.w3_BRN) + self.b3_BRN)

        # Loss, Optimizer and Predictions
        self.cross_entropy_BRN = -tf.reduce_sum(y_*tf.log(self.y_BRN))

        self.train_step_BRN = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy_BRN, global_step = self.steps)

        self.correct_prediction_BRN = tf.equal(tf.arg_max(self.y_BRN,1),tf.arg_max(y_,1))
        self.accuracy_BRN = tf.reduce_mean(tf.cast(self.correct_prediction_BRN, tf.float32))




    # this is a simpler version of Tensorflow's 'official' version. See:
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L102
    def batch_renorm_wrapper(self, inputs, is_training, decay = 0.99):
        # Small epsilon value for the BN transform
        epsilon = 1e-3
        
        def clip_tight(): return tf.constant(1.0), tf.constant(0.0)
        def clip_relax(): return tf.constant(3.0), tf.constant(5.0)
        rmax, dmax = tf.cond(self.steps < tf.constant(25000), clip_tight, clip_relax)
    
        self.scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        self.beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        self.pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        self.pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    
        def training():
            self.batch_mean, self.batch_var = tf.nn.moments(inputs,[0])
            self.train_mean = tf.assign(self.pop_mean,
                                   self.pop_mean * decay + self.batch_mean * (1 - decay))
            self.train_var = tf.assign(self.pop_var,
                                   self.pop_var  * decay + self.batch_var  * (1 - decay))
            with tf.control_dependencies([self.train_mean, self.train_var]):
                pop_sigma = tf.sqrt(self.pop_var, 'sigma')
                r = tf.stop_gradient(tf.clip_by_value(tf.sqrt(self.batch_var / self.pop_var), 1.0 / rmax, rmax))
                d = tf.stop_gradient(tf.clip_by_value((self.batch_mean - self.pop_mean) / pop_sigma, -dmax, dmax))
                x_norm = (inputs - self.batch_mean) / tf.sqrt(self.batch_var + epsilon)
                x_hat = r * x_norm + d
                x_renorm = self.scale * x_hat + self.beta
                return x_renorm
        
        def testing():
            return tf.nn.batch_normalization(inputs,
                self.pop_mean, self.pop_var, self.beta, self.scale, epsilon)

        return tf.cond(is_training, training, testing)

    
