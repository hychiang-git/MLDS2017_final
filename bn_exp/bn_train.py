import numpy as np
import tensorflow as tf, tqdm
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import nets


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)




def run_mnist():
    
    # Placeholders
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    is_training = tf.placeholder(tf.bool, shape=[])

    #mn_ex = nets.MNIST_EX(x, y_)
    mn = nets.MNIST(x, y_)
    mn_bn = nets.MNIST_BN(x, y_)
    mn_bav = nets.MNIST_BN_MVAVG(x, y_, is_training)
    mn_brn = nets.MNIST_BRN(x, y_, is_training)

    zs, BNs, BAVs, BRNs = [], [], [], []
    zsvar, BNsvar, BAVsvar, BRNsvar = [], [], [], []
    acc, acc_BN, acc_BAV, acc_BRN = [], [], [], []
    acc_sin, acc_BN_sin, acc_BAV_sin, acc_BRN_sin = [], [], [], []
    acc_batch, acc_BN_batch, acc_BAV_batch, acc_BRN_batch = [], [], [], []
    
    sess = tf.InteractiveSession()

    '''
        TRAINING
    '''
    sess.run(tf.global_variables_initializer())
    for i in tqdm.tqdm(range(40001)):
        batch = mnist.train.next_batch(60)
        mn.train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        mn_bn.train_step_BN.run(feed_dict={x: batch[0], y_: batch[1]})
        mn_bav.train_step_BAV.run(feed_dict={x: batch[0], y_: batch[1], is_training:True})
        mn_brn.train_step_BRN.run(feed_dict={x: batch[0], y_: batch[1], is_training:True})
        if i % 100 is 0 and i>0:
            '''
                TESTING
            '''
            res = sess.run([mn.accuracy, mn_bn.accuracy_BN, mn_bav.accuracy_BAV, mn_brn.accuracy_BRN, mn.z2, mn_bn.BN2, mn_bav.BAV2, mn_brn.BRN2],
                           feed_dict={x: mnist.test.images, y_: mnist.test.labels, is_training:False})
            acc.append(res[0])
            acc_BN.append(res[1])
            acc_BAV.append(res[2])
            acc_BRN.append(res[3])
            
            zs.append(np.mean(res[4],axis=0)) # record the mean value of z2 over the entire test set
            zsvar.append(np.var(res[4],axis=0)) # record the mean value of z2 over the entire test set
            
            BNs.append(np.mean(res[5],axis=0)) # record the mean value of BN2 over the entire test set
            BNsvar.append(np.var(res[5],axis=0)) # record the mean value of BN2 over the entire test set
            
            BAVs.append(np.mean(res[6],axis=0)) # record the mean value of BN2 over the entire test set
            BAVsvar.append(np.var(res[6],axis=0)) # record the mean value of BN2 over the entire test set
        
            BRNs.append(np.mean(res[7],axis=0)) # record the mean value of BN2 over the entire test set
            BRNsvar.append(np.var(res[7],axis=0)) # record the mean value of BN2 over the entire test set

        #if i % 100 is 0  and i>0:
        #    '''
        #        TESTING batch size 
        #    '''
        #    correct = 0
        #    correct_BN = 0
        #    correct_BAV = 0
        #    correct_BRN = 0
        #    epochs = 1
        #    for i in range(epochs):
        #        batch = mnist.test.next_batch(100)
        #        corr = sess.run([mn.accuracy, mn_bn.accuracy_BN, mn_bav.accuracy_BAV, mn_brn.accuracy_BRN],
        #                        feed_dict={x: batch[0], y_: batch[1], is_training:False})
        #        correct += corr[0]
        #        correct_BN  += corr[1]
        #        correct_BAV += corr[2]
        #        correct_BRN += corr[3]
        #    acc_batch.append(correct/epochs)
        #    acc_BN_batch.append(correct_BN/epochs)
        #    acc_BAV_batch.append(correct_BAV/epochs)
        #    acc_BRN_batch.append(correct_BRN/epochs)
        #if i % 100 is 0  and i>0:
        #    '''
        #        TESTING by single example
        #    '''
        #    correct = 0.
        #    correct_BN = 0.
        #    correct_BAV = 0.
        #    correct_BRN = 0
        #    n_examples = 100
        #    for j in range(n_examples):
        #        corr = sess.run([mn.accuracy, mn_bn.accuracy_BN, mn_bav.accuracy_BAV, mn_brn.accuracy_BRN],
        #                        feed_dict={x: [mnist.test.images[j]], y_: [mnist.test.labels[j]], is_training:False})
        #        correct += corr[0]
        #        correct_BN  += corr[1]
        #        correct_BAV += corr[2]
        #        correct_BRN += corr[3]
        #    acc_sin.append(correct/n_examples)
        #    acc_BN_sin.append(correct_BN/n_examples)
        #    acc_BAV_sin.append(correct_BAV/n_examples)
        #    acc_BRN_sin.append(correct_BRN/n_examples)
    

    
    acc, acc_sin, acc_batch, zs, zsvar   = np.array(acc), np.array(acc_sin), np.array(acc_batch), np.array(zs), np.array(zsvar)
    acc_BN, acc_BN_sin, acc_BN_batch, BNs, BNsvar = np.array(acc_BN), np.array(acc_BN_sin), np.array(acc_BN_batch), np.array(BNs), np.array(BNsvar)
    acc_BAV, acc_BAV_sin, acc_BAV_batch, BAVs, BAVsvar = np.array(acc_BAV), np.array(acc_BAV_sin), np.array(acc_BAV_batch), np.array(BAVs), np.array(BAVsvar)
    acc_BRN, acc_BRN_sin, acc_BRN_batch, BRNs, BRNsvar = np.array(acc_BRN), np.array(acc_BRN_sin), np.array(acc_BRN_batch), np.array(BRNs), np.array(BRNsvar)
    
   
    '''
        TESTING ACCURACY
    ''' 
    fig, ax = plt.subplots()
    ax.plot(range(0,len(acc)*100,100),acc, label='Without BN')
    ax.plot(range(0,len(acc)*100,100),acc_BN, label='With BN')
    ax.plot(range(0,len(acc)*100,100),acc_BAV, label='With BN MAVG')
    ax.plot(range(0,len(acc)*100,100),acc_BRN, label='With BRN')
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0,1])
    ax.set_title('Batch Normalization Accuracy')
    ax.legend(loc=4)
    plt.show()
    
    #'''
    #    TESTING batch examples ACCURACY
    #''' 
    #fig, ax = plt.subplots()
    #ax.plot(range(0,len(acc_batch)*100,100),acc_batch, label='Without BN')
    #ax.plot(range(0,len(acc_batch)*100,100),acc_BN_batch, label='With BN')
    #ax.plot(range(0,len(acc_batch)*100,100),acc_BAV_batch, label='With BN MAVG')
    #ax.plot(range(0,len(acc_batch)*100,100),acc_BRN_batch, label='With BRN')
    #ax.set_xlabel('Training steps')
    #ax.set_ylabel('Accuracy')
    #ax.set_ylim([0,1])
    #ax.set_title('Batch Normalization Accuracy')
    #ax.legend(loc=4)
    #plt.show()


    #'''
    #    TESTING one example ACCURACY
    #''' 
    #fig, ax = plt.subplots()
    #ax.plot(range(0,len(acc_sin)*100,100),acc_sin, label='Without BN')
    #ax.plot(range(0,len(acc_sin)*100,100),acc_BN_sin, label='With BN')
    #ax.plot(range(0,len(acc_sin)*100,100),acc_BAV_sin, label='With BN MAVG')
    #ax.plot(range(0,len(acc_sin)*100,100),acc_BRN_sin, label='With BRN')
    #ax.set_xlabel('Training steps')
    #ax.set_ylabel('Accuracy')
    #ax.set_ylim([0,1])
    #ax.set_title('Batch Normalization Accuracy')
    #ax.legend(loc=4)
    #plt.show()
    
    '''
       NEURON MEAN VAR 
    ''' 
    fig, axes = plt.subplots(5, 2, figsize=(6,12))
    fig.tight_layout()
    for i, ax in enumerate(axes):
        ax[0].set_title("Mean: Without BN")
        ax[1].set_title("Var: Without BN")
        ax[0].plot(zs[:,i])
        ax[1].plot(zsvar[:,i])
    plt.show()

    fig, axes = plt.subplots(5, 2, figsize=(6,12))
    fig.tight_layout()
    for i, ax in enumerate(axes):
        ax[0].set_title("Mean: With BN")
        ax[1].set_title("Var: With BN")
        ax[0].plot(BNs[:,i])
        ax[1].plot(BNsvar[:,i])
    plt.show()
    
    fig, axes = plt.subplots(5, 2, figsize=(6,12))
    fig.tight_layout()
    for i, ax in enumerate(axes):
        ax[0].set_title("Mean: With BN MVAVG")
        ax[1].set_title("Var: With BN MVAVG")
        ax[0].plot(BAVs[:,i])
        ax[1].plot(BAVsvar[:,i])
    plt.show()

    fig, axes = plt.subplots(5, 2, figsize=(6,12))
    fig.tight_layout()
    for i, ax in enumerate(axes):
        ax[0].set_title("Mean: With BRN")
        ax[1].set_title("Var: With BRN")
        ax[0].plot(BRNs[:,i])
        ax[1].plot(BRNsvar[:,i])
    plt.show()


    '''
        TESTING single example
    '''
    correct = 0
    correct_BN = 0
    correct_BAV = 0
    correct_BRN = 0
    n_examples = 1000
    for i in range(n_examples):
        corr = sess.run([mn.accuracy, mn_bn.accuracy_BN, mn_bav.accuracy_BAV, mn_brn.accuracy_BRN],
                        feed_dict={x: [mnist.test.images[i]], y_: [mnist.test.labels[i]], is_training:False})
        correct += corr[0]
        correct_BN  += corr[1]
        correct_BAV += corr[2]
        correct_BRN += corr[3]
    print("ACCURACY:", correct/n_examples)
    print("BN ACCURACY:", correct_BN/n_examples)
    print("BN MVAVG ACCURACY:", correct_BAV/n_examples)
    print("BN RN ACCURACY:", correct_BRN/n_examples)


    '''
        TESTING batch size 
    '''
    correct = 0
    correct_BN = 0
    correct_BAV = 0
    correct_BRN = 0
    epochs = 10
    for i in range(epochs):
        batch = mnist.test.next_batch(100)
        corr = sess.run([mn.accuracy, mn_bn.accuracy_BN, mn_bav.accuracy_BAV, mn_brn.accuracy_BRN],
                        feed_dict={x: batch[0], y_: batch[1], is_training:False})
        correct += corr[0]
        correct_BN  += corr[1]
        correct_BAV += corr[2]
        correct_BRN += corr[3]
    print("ACCURACY:", correct/epochs)
    print("BN ACCURACY:", correct_BN/epochs)
    print("BN MVAVG ACCURACY:", correct_BAV/epochs)
    print("BN RN ACCURACY:", correct_BRN/epochs)

if __name__=="__main__":
    run_mnist()



#            fig, axes = plt.subplots(1, 2, figsize=(6,12))
#            fig.tight_layout()
#            #pca = PCA(n_components=2)
#            #z = pca.fit(res[2]).transform(res[2])
#            model_z = TSNE(n_components=2, random_state=0)
#            z = model_z.fit_transform(res[2])
#            axes[0].set_title("Without BN")
#            axes[0].axis([-50, 50, -50, 50])
#            axes[0].plot(z[:,0], z[:, 1], 'ro')
#            
#            #pca = PCA(n_components=2)
#            #bn = pca.fit(res[3]).transform(res[3])
#            model_bn = TSNE(n_components=2, random_state=0)
#            bn = model_bn.fit_transform(res[3])
#            axes[1].set_title("With BN")
#            axes[1].axis([-50, 50, -50, 50])
#            axes[1].plot(bn[:, 0], bn[:, 1], 'ro')
#            plt.show()
