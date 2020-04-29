#1.downlowd the database of images from mnist website
import numpy as np
from numpy import reshape
import os
def load_dataset():
    def download(filename,source="http://yann.lecun.com/exdb/mnist/"):
        print("Downloading",filename)
        import urllib
        #from urllib import urlretrieve
        urllib.request.urlretrieve(source+filename,filename)
        #this will download the file to local memory
    import gzip
    
    def load_mnist_images(filename):
        #checks if file is already there on loaal disk or not if not it will download
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename,'rb') as f:
            #open the zip file of images
            data=np.frombuffer(f.read(), np.uint8, offset=16)
            #this is some boilerplate to extract data from zip file
            #converting array into images. Each image has 28x28 pixes,its a monochrome image so only 1 channel(using reshape)
            data=data.reshape(-1,1,28,28)
            #1st-no of images
            #2nd-no of channels
            #3,4th-pixels
        return data/np.float32(256)#this will convert byte to float
    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename,'rb') as f:
            #open the zip file of images
            data=np.frombuffer(f.read(), np.uint8, offset=16)
            #data=data.reshape(-1,1,28,28)
        return data
    x_train=load_mnist_images('train-images-idx3-ubyte.gz')
    y_train=load_mnist_labels('train-labels-idx1-ubyte.gz')
    x_test=load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test=load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    return x_train,y_train,x_test,y_test
xtr,ytr,xtes,ytes=load_dataset()
#2.set up a nueral network with req no of layers and node
#train the network
#3.feed in the training data
#4.how the output is for 1 image
#5.feedind a test data set to trained nueral network

#we've got the data now we want to test it using one of the images using matplotlib
import matplotlib
matplotlib.use('TkAgg')#this is setting for matplotlib to render images

import matplotlib.pyplot as plt
plt.show(plt.imshow(xtr[0][0]))


#we are going to use 2 python packages-theano and lasagne
#theano-is a mathematical package that allows u to define and perform mathematical computations just like numpy but at
#higher dimension arrays and these are called tensors
#lasagne ia a library that is built on theono to build nueral networks. it comes with functions to set up layers,define error functions
#and how the training of nueral network would be
import lasagne
from lasagne.utils import int_types
import theano
import theano.tensor as T
def build_NN(input_var=None):
    #we are going to build a nueral network with 2 hidden layers of 800 nodes each the o/p layer will have 10 node
    #ranging from 0-9 the node with max value is the predicted output
    
    #input layer has 784 nodes that is of shape 1x28x28 (1-channel)
    l_in=lasagne.layers.InputLayer(shape=(None,1,28,28),input_var=input_var)
    #there will be 20% dropout-to avoid overfitting
    l_in_drop=lasagne.layers.DropoutLayer(l_in,p=0.2)
    #add alayer with 800 nodes. Initially this will be dense/fully-connected
    l_hid1=lasagne.layers.DenseLayer(l_in_drop,num_units=800,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    #dropout layer
    l_hid1_drop=lasagne.layers.DropoutLayer(l_hid1,p=0.5)
    l_hid2=lasagne.layers.DenseLayer(l_hid1_drop,num_units=800,nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    l_hid2_drop=lasagne.layers.DropoutLayer(l_hid2 ,p=0.5)
    l_out=lasagne.layers.DenseLayer(l_hid2_drop,num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
    return l_out#we return the final output

#we have setup the network. now we tell to train the network how to train itself

input_var =T.tensor4('inputs')#empty 4 dimensional array
target_var=T.ivector('targets')#an empty 1 dimensional integer array tp repr the labels
network=build_NN(input_var)
#training
#1.compute error function
prediction=lasagne.layers.get_output(network)
loss=lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss=loss.mean()

#2.updation of weights
params=lasagne.layers.get_all_params(network, trainable=True)
updates=lasagne.updates.nesterov_momentum(loss,params,learning_rate=0.01,momentum=0.9)

#single training step
train_fn=theano.function([input_var,target_var],loss,updates=updates)


#step3
num_training_steps=10
 
for step in range(num_training_steps):
    train_err=train_fn(xtr,ytr)
    print("current step is"+str(step))
    
    
 #check with 1 image
test_prediction=lasagne.layers.get_output(network)
val_fn=theano.function([input_var],test_prediction)

val_fn([xtes[0]])

ytes[0]

test_prediction=lasagne.layers.get_output(network)
test_acc=T.mean(T.eq(T.argmax(test_prediction,axis=1),target_var),dtype=theano.config.floatX)

acc_fn=theano.function([input_var,target_var],test_acc)
acc_fn(xtes,ytes)
