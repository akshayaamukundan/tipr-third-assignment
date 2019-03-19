import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle
import random
import math
import operator
# import tensorlayer as tl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


# dataset = 'CIFAR-10'
# batchnum = 1

class image:
    def __init__(self, dataset):
        self.dataset = dataset
        if (dataset == 'CIFAR-10'):
            self.numchannel = 3
            self.numrows = 32
            self.numcols = 32
            self.numclass = 10
            self.labeldict = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        if (dataset == 'Fashion-MNIST'):
            self.numchannel = 1
            self.numrows = 28
            self.numcols = 28
            self.numclass = 10
            self.labeldict = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker','Bag', 'Ankle boot']

    def datasetextract(self, batchnum):
        if (self.dataset == 'CIFAR-10'):
            # file = "..\\data\\CIFAR-10\\data_batch_1"  # format different in clserver
            # /storage2/home2/e1-313-15521/tipr-third-assignment/data/
            file = '../data/CIFAR-10' + '/data_batch_' + str(batchnum)
            #file = '/storage2/home2/e1-313-15521/tipr-third-assignment/data/CIFAR-10' + '/data_batch_' + str(batchnum)
            picklein = open(file, "rb")
            X1 = pickle.load(picklein, encoding='latin1')
            self.data = X1["data"].reshape(-1, 32, 32, 3)  # not normalized
            self.label = X1["labels"]
        if (self.dataset == 'Fashion-MNIST'):
            file = '../data/Fashion-MNIST'
            datafmnist = input_data.read_data_sets(file, one_hot=True)
            #datafmnist = input_data.read_data_sets('/storage2/home2/e1-313-15521/tipr-third-assignment/data/Fashion-MNIST', one_hot=True)
            self.data = np.zeros((len(datafmnist.train.images), 28, 28, 1))
            for i in range(len(datafmnist.train.images)):
                self.data[i] = datafmnist.train.images[i].reshape(28, 28, 1)
            self.label = datafmnist.train.labels

    def labelexpand(self):
        if (self.dataset == 'CIFAR-10'):
            labelset = np.zeros((len(self.label), 10))
            for i in range(len(self.label)):
                labelset[i] = np.eye(10)[self.label[i]]
        if (self.dataset == 'Fashion-MNIST'):
            labelset = self.label
        return labelset


def myconv(dataset, activation, numneurons, inputs,
           channels):

    if (len(numneurons) < 2):
        print('Number of layers must be atleast 3!')
        exit(0)
    # put this at the starting of code
    prevlayerneurons = channels
    nextlayerneurons = numneurons[0]
    layerout = inputs
    for l in range(len(numneurons)):  # 2 is the number of fully connected layers

        kernel = tf.Variable(tf.truncated_normal(shape=[3, 3, prevlayerneurons, nextlayerneurons], mean=0, stddev=0.08))
        # kernel = tf.get_variable('W' + str(l), shape=(3, 3, prevlayerneurons, nextlayerneurons),initializer=tf.contrib.layers.xavier_initializer())
        strides = 1
        convlayerout = tf.nn.conv2d(layerout, kernel, strides=[1, strides, strides, 1], padding='SAME')

        if (activation == 'relu'):
            convlayerout = tf.nn.relu(convlayerout)
        elif (activation == 'sigmoid'):

            convlayerout = tf.nn.sigmoid(convlayerout)
        elif (activation == 'tanh'):

            convlayerout = tf.nn.tanh(convlayerout)
        elif (activation == 'swish'):
            convlayerout = tf.nn.swish(convlayerout)
        else:
            print('nil')

        filtersize = 2

        maxpoollayerout = tf.nn.max_pool(convlayerout, ksize=[1, filtersize, filtersize, 1],
                                         strides=[1, filtersize, filtersize, 1], padding='SAME')
        normalizedlayerout = tf.layers.batch_normalization(maxpoollayerout)  # check if this works
        prevlayerneurons = nextlayerneurons
        nextlayerneurons = numneurons[l]
        layerout = normalizedlayerout
    ############## conv layer ends //check if we have to find number of conv layers, modify loop accordingly

    flattenedlayerout = tf.contrib.layers.flatten(layerout)

    fullyconnectedlayerout1 = tf.contrib.layers.fully_connected(inputs=flattenedlayerout, num_outputs=64,
                                                                activation_fn=tf.nn.relu)


    fullyconnectedlayerout1 = tf.layers.batch_normalization(fullyconnectedlayerout1)
    outofnn = tf.contrib.layers.fully_connected(inputs=fullyconnectedlayerout1, num_outputs=10, activation_fn=None)

    return outofnn, fullyconnectedlayerout1


def normalizingdata(myobject):
    data = myobject.data
    mindatavalue = np.min(data)
    maxdatavalue = np.max(data)
    data = data / maxdatavalue  # check this
    return data


def clusteringaccuracy(obtainedlabel, actuallabel):
    clusteraccuracy = 0
    for n in range(10):
        clusteraccuracynew = 0
        for i in range(len(obtainedlabel)):
            print(obtainedlabel[i])
            print(actuallabel[i])
            if (np.mod((obtainedlabel[i]+n),10) == (actuallabel[i])):
                clusteraccuracynew += 1
        clusteraccuracynew = clusteraccuracynew / float(len(obtainedlabel))
        if (clusteraccuracynew > clusteraccuracy):
            clusteraccuracy = clusteraccuracynew
        else:
            pass
    return clusteraccuracy


if __name__ == "__main__":
    config = tf.ConfigProto(intra_op_parallelism_threads=0,
                            inter_op_parallelism_threads=0,
                            allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    epochnum = 1
    train = None  # 'yes'  #
    batch_size = 128
    learning_rate = 0.001
    activation = 'tanh'  # change to ReLu
    datasetName = 'Fashion-MNIST'  # 'CIFAR-10' #
    myobject = image(datasetName)  # ?
    # numNodes = [64, 128, 256, 512, 128, 256]
    numneurons = [64, 64]

    tf.reset_default_graph()

    inputs = tf.placeholder(tf.float32, shape=(None, myobject.numrows, myobject.numcols, myobject.numchannel),
                            name='input_x')
    target = tf.placeholder(tf.float32, shape=(None, myobject.numclass), name='output_y')

    pred, outputlayer = myconv(datasetName, activation, numneurons, inputs, myobject.numchannel)
    # dataset, activation, numneurons, inputs, channels
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=target))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    #save_model_path = '/storage2/home2/e1-313-15521/tipr-third-assignment/Models/' + datasetName
    #save_model_path = '../Models/' + datasetName
    if (train != None):

        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=0,
                                              inter_op_parallelism_threads=1)) as sess:
            sess.run(tf.global_variables_initializer())
            batchnum = 5  # loop it at the end
            testSet = []
            labeltest = []
            for b in range(1, batchnum + 1):  # check if plus 1 required or not
                myobject.datasetextract(b)
                data = normalizingdata(myobject)
                labelexp = myobject.labelexpand()
                xtrain, xtest, ltrain, ltest = train_test_split(data, labelexp,
                                                                test_size=0.9)  # make it to zero at last
                testSet.append(xtest)
                labeltest.append(ltest)
                for epoch in range(epochnum):
                    print('epoch:', epoch)
                    for j in range(0, len(xtrain), batch_size):
                        end = min(j + batch_size, len(data))  # why?

                        sess.run(optimizer, feed_dict={inputs: xtrain[j:end], target: ltrain[j:end]})
                        # obtainedcost = sess.run(cost, feed_dict={inputs: data[j:end],target:labelexp[j:end]})
                        # obtainedcostaccuracy = sess.run(accuracy, feed_dict={inputs: data[j:end],target:labelexp[j:end]})

                        # print('Cost::{:>10.4f} Accuracy:: {:.6f}'.format(obtainedcost,obtainedcostaccuracy))
            embeddings = sess.run(outputlayer, feed_dict={inputs: testSet[0]})  # check if it works with xtest alone
            pickleout = open("Xfmnistemb.pickle", "wb")
            pickle.dump(embeddings, pickleout)
            pickleout.close()
            kmeans = KMeans(n_clusters=10).fit(embeddings)
            obtainedclass = kmeans.predict(embeddings)
            clusteraccuracy = clusteringaccuracy(obtainedclass, np.argmax(labeltest[0], axis=1))
            print("Clustering accuracy::", clusteraccuracy)

            tsne = TSNE(n_components=2)

            X_2d = tsne.fit_transform(embeddings)
            target_ids = range(10)

            y = labeltest[0]
            pickleout = open("Xfmnistlabel.pickle", "wb")
            pickle.dump(labeltest[0], pickleout)
            pickleout.close()
            labels = np.argmax(labeltest[0], axis=1)
            plt.figure(figsize=(6, 5))
            # colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'
            print(X_2d[:, 0])
            print(X_2d[:, 1])
            plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels)
            plt.savefig('tsneFashionmnist')
            plt.close()

