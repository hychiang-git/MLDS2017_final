import numpy as np
import tensorflow as tf, tqdm
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import nets

from scipy.linalg import norm
from scipy.spatial.distance import euclidean




mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

_SQRT2 = np.sqrt(2)     # sqrt(2) with default precision np.float64


def fl2(p, q):
    return np.linalg.norm(p-q)

def l2(p, q):
    d = np.linalg.norm(p-q, ord=2, axis=1)
    return np.sum(d)


def hellinger1(p, q):
    return norm(np.sqrt(p) - np.sqrt(q)) / _SQRT2


def hellinger2(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2


def hellinger3(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2



def hellinger_square_multivar(mu1, cov1, mu2, cov2):
    d1_ss = np.sqrt(np.sqrt(np.linalg.det(cov1)))
    d2_ss = np.sqrt(np.sqrt(np.linalg.det(cov2)))
    d12_s = np.sqrt(np.linalg.det((cov1 + cov2)/2.))
    d12_inv = np.linalg.inv((cov1+cov2)/2.)
    dmu = mu1-mu2
    
    a = d1_ss * d2_ss / d12_s
    sigma = (-1./8.)*np.dot(dmu, np.dot(d12_inv, dmu))
    print "a:", a
    print "sigma:", sigma
    #import IPython
    #IPython.embed()
    
    return 1-a*np.exp(sigma)
    

def kl(mu1, cov1, mu2, cov2):
    #return np.sum(np.where(p!=0, p* np.log(p/q), 0))
    #from scipy.stats import entropy
    #return entropy(p, q)
    cov2_inv = np.linalg.inv(cov2)
    dmu = mu1-mu2
    det1 = np.linalg.det(cov1)
    det2 = np.linalg.det(cov2)
    sigma = np.dot(dmu, np.dot(cov2_inv, dmu))
    return (np.trace(np.dot(cov2_inv, cov1)) + sigma - len(mu1) + np.log(det2/ det1))/2.0
    

def js(M1, mu1, cov1, M2, mu2, cov2):
    M = (M1 + M2)/2.0
    M_cov = np.cov(M, rowvar=False)
    M_mean = np.mean(M, axis=0)
    kl_M1_M = kl(mu1, cov1, M_mean, M_cov)
    kl_M2_M = kl(mu2, cov2, M_mean, M_cov)
    return (kl_M1_M + kl_M2_M) * 0.5



def run_mnist():
    
    # Placeholders
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    is_training = tf.placeholder(tf.bool, shape=[])

    mn = nets.MNIST(x, y_)
    mn_bn = nets.MNIST_BN(x, y_)
    mn_bav = nets.MNIST_BN_MVAVG(x, y_, is_training)
    mn_brn = nets.MNIST_BRN(x, y_, is_training)
  
    z1, z1_mean, z1_cov = [], [], []
    z2, z2_mean, z2_cov = [], [], []
    BN1, BN1_mean, BN1_cov = [], [], []
    BN2, BN2_mean, BN2_cov = [], [], []


    sess = tf.InteractiveSession()
    '''
        TRAINING
    '''
    sess.run(tf.global_variables_initializer())

    for i in tqdm.tqdm(range(40001)):
        batch = mnist.train.next_batch(60)
        mn.train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        mn_bn.train_step_BN.run(feed_dict={x: batch[0], y_: batch[1]})
        #mn_bav.train_step_BAV.run(feed_dict={x: batch[0], y_: batch[1], is_training:True})
        #mn_brn.train_step_BRN.run(feed_dict={x: batch[0], y_: batch[1], is_training:True})
        if i % 100 is 0 and i>0:
            '''
                TESTING
            '''
            labels = np.argmax(mnist.test.labels, axis=1)
            zero_id = np.where(labels==0)
            labels = mnist.test.labels[zero_id]
            images = mnist.test.images[zero_id]
            
            res = sess.run([mn.accuracy, mn_bn.accuracy_BN, mn.z1, mn.z2, mn_bn.BN1, mn_bn.BN2],
                           feed_dict={x: images, y_: labels, is_training:False})
            #res = sess.run([mn.accuracy, mn_bn.accuracy_BN, mn.z2, mn_bn.BN2],
            #               feed_dict={x: mnist.test.images, y_: mnist.test.labels, is_training:False})
            #res = sess.run([mn.accuracy, mn_bn.accuracy_BN, mn_bav.accuracy_BAV, mn_brn.accuracy_BRN, mn.z2, mn_bn.BN2, mn_bav.BAV2, mn_brn.BRN2],
            #               feed_dict={x: mnist.test.images, y_: mnist.test.labels, is_training:False})
            #res = sess.run([mn.accuracy, mn.z2],
            #               feed_dict={x: mnist.test.images, y_: mnist.test.labels, is_training:False})
            z1.append(res[2]) # record the mean value of z2 over the entire test set
            z1_mean.append( np.mean(res[2], axis=0)) # record the mean value of z2 over the entire test set
            z1_cov.append(np.cov(res[2], rowvar=False)) # record the mean value of z2 over the entire test set
            z2.append(res[3]) # record the mean value of z2 over the entire test set
            z2_mean.append( np.mean(res[3], axis=0)) # record the mean value of z2 over the entire test set
            z2_cov.append(np.cov(res[3], rowvar=False)) # record the mean value of z2 over the entire test set

            BN1.append(res[4]) # record the mean value of BN2 over the entire test set
            BN1_mean.append(np.mean(res[4], axis=0)) # record the mean value of BN2 over the entire test set
            BN1_cov.append(np.cov(res[4], rowvar=False)) # record the mean value of BN2 over the entire test set
            BN2.append(res[5]) # record the mean value of BN2 over the entire test set
            BN2_mean.append(np.mean(res[5], axis=0)) # record the mean value of BN2 over the entire test set
            BN2_cov.append(np.cov(res[5], rowvar=False)) # record the mean value of BN2 over the entire test set


    z1 = np.array(z1)
    z1_mean = np.array(z1_mean)
    z1_cov = np.array(z1_cov)
    z2 = np.array(z2)
    z2_mean = np.array(z2_mean)
    z2_cov = np.array(z2_cov)

    BN1 = np.array(BN1)
    BN1_mean = np.array(BN1_mean)
    BN1_cov = np.array(BN1_cov)
    BN2 = np.array(BN2)
    BN2_mean = np.array(BN2_mean)
    BN2_cov = np.array(BN2_cov)

    #z1_js , z2_js = [], []
    #BN1_js , BN2_js = [], []
    #print "-------------------------Z1-------------------------"
    #for i in range(len(z1)-1):
    #    #print "KL: ", kl(z1_mean[i], z1_cov[i], z1_mean[i+1], z1_cov[i+1])
    #    #print "KL: ", kl(z1_mean[i+1], z1_cov[i+1], z1_mean[i], z1_cov[i])
    #    #print "JS: ", js(z1[i], z1_mean[i], z1_cov[i], z1[i+1], z1_mean[i+1], z1_cov[i+1])
    #    #print "JS: ", js(z1[i], z1_mean[i], z1_cov[i], z1[i+1], z1_mean[i+1], z1_cov[i+1])
    #    z1_js.append(js(z1[i+1], z1_mean[i+1], z1_cov[i+1], z1[i], z1_mean[i], z1_cov[i]))
    #print "-------------------------Z2-------------------------"
    #for i in range(len(z2)-1):
    #    #print "KL: ", kl(z2_mean[i], z2_cov[i], z2_mean[i+1], z2_cov[i+1])
    #    #print "KL: ", kl(z2_mean[i+1], z2_cov[i+1], z2_mean[i], z2_cov[i])
    #    #print "JS: ", js(z2[i], z2_mean[i], z2_cov[i], z2[i+1], z2_mean[i+1], z2_cov[i+1])
    #    z2_js.append(js(z2[i+1], z2_mean[i+1], z2_cov[i+1], z2[i], z2_mean[i], z2_cov[i]))
    #print "------------------------BN1------------------------------"
    #for i in range(len(BN1)-1):
    #    print js(BN1[i], BN1_mean[i], BN1_cov[i], BN1[i+1], BN1_mean[i+1], BN1_cov[i+1])
    #    #BN1_js.append(js(BN1[i], BN1_mean[i], BN1_cov[i], BN1[i+1], BN1_mean[i+1], BN1_cov[i+1]))
    #print "------------------------BN2-------------------------------"
    #for i in range(len(BN2)-1):
    #    print js(BN2[i], BN2_mean[i], BN2_cov[i], BN2[i+1], BN2_mean[i+1], BN2_cov[i+1])
    #    #BN2_js.append(js(BN2[i], BN2_mean[i], BN2_cov[i], BN2[i+1], BN2_mean[i+1], BN2_cov[i+1]))

    #fig, ax = plt.subplots()
    #ax.plot(range(0,len(z1_js)*100,100), z1_js, label='z1:Without BN')
    ##ax.plot(range(0,len(BN1_js)*100,100),BN1_js, label='z1:BN')
    ##ax.plot(range(0,len(BN2_js)*100,100),BN2_js, label='z2:BN')
    #ax.legend(loc=0)
    #plt.ylabel('JS Distance')
    #plt.xlabel('Training step')
    #plt.show()

    #fig, ax = plt.subplots()
    #ax.plot(range(0,len(z2_js)*100,100), z2_js, label='z2:Without BN')
    ##ax.plot(range(0,len(BN1_js)*100,100),BN1_js, label='z1:BN')
    ##ax.plot(range(0,len(BN2_js)*100,100),BN2_js, label='z2:BN')
    #ax.legend(loc=0)
    #plt.ylabel('JS Distance')
    #plt.xlabel('Training step')
    #plt.show()

    #BAVs = np.array(BAVs)
    #BRNs = np.array(BRNs)
    #print z1.shape
    #print z1_mean.shape
    #print z1_cov.shape
    #print z2.shape
    #print z2_mean.shape
    #print z2_cov.shape
    #print BN1.shape
    #print BN1_mean.shape
    #print BN1_cov.shape
    #print BN2.shape
    #print BN2_mean.shape
    #print BN2_cov.shape

    #z1_h , z2_h = [], []
    #BN1_h , BN2_h = [], []
    #print "-------------------------Z1-------------------------"
    #for i in range(len(z1)-1):
    #    #print hellinger_square_multivar(zsmean[i], zsvar[i], zsmean[i+1], zsvar[i+1])
    #    z1_h.append(hellinger1(abs(z1[i]), abs(z1[i+1])))
    #    #print hellinger3(abs(z1[i]), abs(z1[i+1]))
    #print "-------------------------Z2-------------------------"
    #for i in range(len(z2)-1):
    #    #print hellinger_square_multivar(zsmean[i], zsvar[i], zsmean[i+1], zsvar[i+1])
    #    z2_h.append(hellinger1(abs(z2[i]), abs(z2[i+1])))
    #    #print hellinger1(abs(z2[i]), abs(z2[i+1]))
    #    #print hellinger3(abs(z2[i]), abs(z2[i+1]))
    #print "------------------------BN1------------------------------"
    #for i in range(len(BN1)-1):
    #    #print hellinger_square_multivar(BNsmean[i], BNsvar[i], BNsmean[i+1], BNsvar[i+1])
    #    BN1_h.append(hellinger1(abs(BN1[i]), abs(BN1[i+1])))
    #    #print hellinger1(abs(BN1[i]), abs(BN1[i+1]))
    #    #print hellinger3(abs(BN1[i]), abs(BN1[i+1]))
    #print "------------------------BN2-------------------------------"
    #for i in range(len(BN2)-1):
    #    #print hellinger_square_multivar(BNsmean[i], BNsvar[i], BNsmean[i+1], BNsvar[i+1])
    #    BN2_h.append(hellinger1(abs(BN2[i]), abs(BN2[i+1])))
    #    #print hellinger1(abs(BN2[i]), abs(BN2[i+1]))
    #    #print hellinger3(abs(BN2[i]), abs(BN2[i+1]))



    #fig, ax = plt.subplots()
    #ax.plot(range(0,len(z1_h)*100,100), z1_h, label='z1:Without BN')
    #ax.plot(range(0,len(z2_h)*100,100), z2_h, label='z2:Without BN')
    #ax.plot(range(0,len(BN1_h)*100,100),BN1_h, label='z1:BN')
    #ax.plot(range(0,len(BN2_h)*100,100),BN2_h, label='z2:BN')
    #ax.legend(loc=0)
    #plt.ylabel('Hellinger Distance')
    #plt.xlabel('Training')
    #plt.show()

    #z1_l2 , z2_l2 = [], []
    #BN1_l2 , BN2_l2 = [], []
    #print "-------------------------Z1-------------------------"
    #for i in range(len(z1)-1):
    #    z1_l2.append(l2(z1[i], z1[i+1]))
    #print "-------------------------Z2-------------------------"
    #for i in range(len(z2)-1):
    #    z2_l2.append(l2(z2[i], z2[i+1]))
    #print "------------------------BN1------------------------------"
    #for i in range(len(BN1)-1):
    #    BN1_l2.append(l2(BN1[i], BN1[i+1]))
    #print "------------------------BN2-------------------------------"
    #for i in range(len(BN2)-1):
    #    BN2_l2.append(l2(BN2[i], BN2[i+1]))



    #fig, ax = plt.subplots()
    #ax.plot(range(0,len(z1_l2)*100,100), z1_l2, label='z1:Without BN')
    #ax.plot(range(0,len(z2_l2)*100,100), z2_l2, label='z2:Without BN')
    #ax.plot(range(0,len(BN1_l2)*100,100),BN1_l2, label='z1:BN')
    #ax.plot(range(0,len(BN2_l2)*100,100),BN2_l2, label='z2:BN')
    #ax.legend(loc=0)
    #plt.ylabel('L2 Distance')
    #plt.xlabel('Training')
    #plt.show()

    z1_fl2 , z2_fl2 = [], []
    BN1_fl2 , BN2_fl2 = [], []
    print "-------------------------Z1-------------------------"
    for i in range(len(z1)-1):
        z1_fl2.append(fl2(z1[i], z1[i+1]))
    print "-------------------------Z2-------------------------"
    for i in range(len(z2)-1):
        z2_fl2.append(fl2(z2[i], z2[i+1]))
    print "------------------------BN1------------------------------"
    for i in range(len(BN1)-1):
        BN1_fl2.append(fl2(BN1[i], BN1[i+1]))
    print "------------------------BN2-------------------------------"
    for i in range(len(BN2)-1):
        BN2_fl2.append(fl2(BN2[i], BN2[i+1]))



    fig, ax = plt.subplots()
    ax.plot(range(0,len(z1_fl2)*100,100), z1_fl2, label='z1:Without BN')
    ax.plot(range(0,len(z2_fl2)*100,100), z2_fl2, label='z2:Without BN')
    ax.plot(range(0,len(BN1_fl2)*100,100),BN1_fl2, label='z1:BN')
    ax.plot(range(0,len(BN2_fl2)*100,100),BN2_fl2, label='z2:BN')
    ax.legend(loc=0)
    plt.ylabel('Frobenius Distance')
    plt.xlabel('Training')
    plt.show()
if __name__=="__main__":
    run_mnist()


