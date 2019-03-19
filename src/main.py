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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from scipy import ndimage
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import argparse


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
            self.labeldict = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
                              'Bag', 'Ankle boot']

    def datasetextract(self, batchnum, trainpath, datasetName):
        if (self.dataset == 'CIFAR-10'):
            # file = "..\\data\\CIFAR-10\\data_batch_1"  # format different in clserver
            # /storage2/home2/e1-313-15521/tipr-third-assignment/data/
            file = trainpath + '/' + datasetName + '/data_batch_' + str(batchnum)
            # file = '/storage2/home2/e1-313-15521/tipr-third-assignment/data/CIFAR-10' + '/data_batch_' + str(batchnum)
            picklein = open(file, "rb")
            X1 = pickle.load(picklein, encoding='latin1')
            self.data = X1["data"].reshape(-1, 32, 32, 3)  # normalized later
            self.label = X1["labels"]
        if (self.dataset == 'Fashion-MNIST'):
            file1 = trainpath + '/' + datasetName
            datafmnist = input_data.read_data_sets(file1, one_hot=True)
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


def myconv(dataset, activation, numneurons, inputs, channels):
    prevlayerneurons = channels
    nextlayerneurons = 64
    layerout = inputs
    for l in range(len(numneurons)):
        if (dataset == 'CIFAR-10'):
            kernel = tf.Variable(
                tf.truncated_normal(shape=[numneurons[l], numneurons[l], prevlayerneurons, nextlayerneurons], mean=0,
                                    stddev=0.1))
        elif (dataset == 'Fashion-MNIST'):
            kernel = tf.get_variable('W' + str(l),
                                     shape=(numneurons[l], numneurons[l], prevlayerneurons, nextlayerneurons),
                                     initializer=tf.contrib.layers.xavier_initializer())
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
        normalizedlayerout = tf.layers.batch_normalization(maxpoollayerout)
        prevlayerneurons = nextlayerneurons
        nextlayerneurons = 64
        layerout = normalizedlayerout
    ############## convolutional layer ends ###################

    flattenedlayerout = tf.contrib.layers.flatten(layerout)
    fullyconnectedlayerout1 = tf.contrib.layers.fully_connected(inputs=flattenedlayerout, num_outputs=(128),
                                                                activation_fn=tf.nn.relu)

    fullyconnectedlayerout1 = tf.layers.batch_normalization(fullyconnectedlayerout1)
    fullyconnectedlayerout2 = tf.contrib.layers.fully_connected(inputs=fullyconnectedlayerout1, num_outputs=64,
                                                                activation_fn=tf.nn.relu)

    fullyconnectedlayerout2 = tf.layers.batch_normalization(fullyconnectedlayerout2)
    outofnn = tf.contrib.layers.fully_connected(inputs=fullyconnectedlayerout2, num_outputs=10, activation_fn=None)
    return outofnn, fullyconnectedlayerout2


def normalizingdata(myobject):
    data = myobject.data
    maxdatavalue = np.max(data)
    data = data / maxdatavalue
    return data


def clusteringaccuracy(obtainedlabel, actuallabel):
    clusteraccuracy = 0
    for n in range(10):
        clusteraccuracynew = 0
        for i in range(len(obtainedlabel)):
            print(obtainedlabel[i])
            print(actuallabel[i])
            if (np.mod((obtainedlabel[i] + n), 10) == (actuallabel[i])):
                clusteraccuracynew += 1
        clusteraccuracynew = clusteraccuracynew / float(len(obtainedlabel))
        if (clusteraccuracynew > clusteraccuracy):
            clusteraccuracy = clusteraccuracynew
        else:
            pass
    return clusteraccuracy


if __name__ == "__main__":
    config = tf.ConfigProto(intra_op_parallelism_threads=0, inter_op_parallelism_threads=0, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", help="train data path", default=None)
    parser.add_argument("--test-data", help="test data path")
    parser.add_argument("--dataset", help="CIFAR-10 or Fashion-MNIST")
    parser.add_argument("--filter-config", help="Filter for each conv layer", nargs="*")
    parser.add_argument("--activation", help="activation for each conv layer")
    args = parser.parse_args()

    config = args.filter_config
    if (config != None):
        numneurons = []
        a = config[0].split('[')
        numneurons.append(int(a[1]))
        for i in range(1, len(config) - 1):
            numneurons.append(int(config[i]))
        b = config[len(config) - 1].split(']')
        numneurons.append(int(b[0]))

    trainpath = args.train_data
    testpath = args.test_data
    datasetName = args.dataset
    activation = args.activation

    if ((trainpath != None) and (len(numneurons) < 2)):
        print('Number of layers must be atleast 2!')
        exit(0)
    if ((datasetName == 'Fashion-MNIST') or (datasetName == 'CIFAR-10')):
        pass
    else:
        print('Name of dataset: Fashion-MNIST or CIFAR-10')
        exit(0)
    if (trainpath != None):
        if ((activation == 'sigmoid') or (activation == 'relu') or (activation == 'tanh') or (activation == 'swish')):
            pass
        else:
            print('activation name: sigmoid or relu or tanh or swish')
            exit(0)

    epochnum = 1
    batch_size = 128
    learning_rate = 0.001

    train = trainpath
    train1 = trainpath

    myobject = image(datasetName)

    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=(None, myobject.numrows, myobject.numcols, myobject.numchannel),
                            name='input_x')
    target = tf.placeholder(tf.float32, shape=(None, myobject.numclass), name='output_y')
    if (train1 != None):
        
        #save_model_path = '/storage2/home2/e1-313-15521/tipr-third-assignment/Models/' + datasetName + '/' + datasetName
        save_model_path = '../Models/' + datasetName + '/' + datasetName

    else:
        #save_model_path = '/storage2/home2/e1-313-15521/tipr-third-assignment/Modelsfinal/' + datasetName + '/' + datasetName
        save_model_path = '../Modelsfinal/' + datasetName

    if (train != None):
        pred, outputlayer = myconv(datasetName, activation, numneurons, inputs, myobject.numchannel)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=target))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(target, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=0,
                                              inter_op_parallelism_threads=1)) as sess:
            sess.run(tf.global_variables_initializer())
            if (datasetName == 'CIFAR-10'):
                batchnum = 5
            elif (datasetName == 'Fashion-MNIST'):
                batchnum = 1
            for epoch in range(epochnum):
                print('epoch:', epoch)
                for b in range(1, batchnum + 1):  # check if plus 1 required or not
                    myobject.datasetextract(b, trainpath, datasetName)
                    data = normalizingdata(myobject)
                    labelexp = myobject.labelexpand()
                    xtrain, xtest, ltrain, ltest = train_test_split(data, labelexp,
                                                                    test_size=0.1)  # make it to zero at last

                    for j in range(0, len(xtrain), batch_size):
                        end = min(j + batch_size, len(data))  # why?

                        sess.run(optimizer, feed_dict={inputs: xtrain[j:end], target: ltrain[j:end]})
                        # obtainedcost = sess.run(cost, feed_dict={inputs: data[j:end], target: labelexp[j:end]})
                        # obtainedcostaccuracy = sess.run(accuracy,feed_dict={inputs: data[j:end], target: labelexp[j:end]})
                        # print('Cost::{:>10.4f} Accuracy:: {:.6f}'.format(obtainedcost, obtainedcostaccuracy))

            prediction = sess.run(pred, feed_dict={inputs: xtest})
            obtainedlabel = np.argmax(prediction, axis=1)
            actuallabel = np.argmax(ltest, axis=1)

            print("Validation test accuracy ::", accuracy_score(actuallabel, obtainedlabel))
            print("Validation test Macro F1 Score::", f1_score(actuallabel, obtainedlabel, average='macro'))
            print("Validation test Micro F1 Score::", f1_score(actuallabel, obtainedlabel, average='micro'))

            saver = tf.train.Saver()
            save_path = saver.save(sess, save_model_path)
    train = None
    if (train == None):
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(save_model_path + '.meta')
        with tf.Session() as sesst:
            # saver.restore(sesst, tf.train.latest_checkpoint('../Models'))
            if (train1 != None):
                #saver.restore(sesst,tf.train.latest_checkpoint('/storage2/home2/e1-313-15521/tipr-third-assignment/Models' + '/' + datasetName))
                saver.restore(sesst,tf.train.latest_checkpoint('../Models' + '/' + datasetName))

            else:
                #saver.restore(sesst,tf.train.latest_checkpoint('/storage2/home2/e1-313-15521/tipr-third-assignment/Modelsfinal' + '/' + datasetName))
                saver.restore(sesst,tf.train.latest_checkpoint('../Modelsfinal' + '/' + datasetName))

            graph = tf.get_default_graph()
            #for operatn in graph.get_operations():
                #print(operatn.name)
            if (datasetName == 'CIFAR-10'):
                batchnum = 5
            elif (datasetName == 'Fashion-MNIST'):
                batchnum = 1
            for b in range(1, batchnum + 1):
                if (datasetName == 'CIFAR-10'):
                    # file = "..\\data\\CIFAR-10\\data_batch_1"  # format different in clserver
                    # /storage2/home2/e1-313-15521/tipr-third-assignment/data/
                    file = testpath + '/' + datasetName # + '/data_batch_' + str(batchnum)
                    # file = '/storage2/home2/e1-313-15521/tipr-third-assignment/data/CIFAR-10' + '/data_batch_' + str(batchnum)
                    picklein = open(file, "rb")
                    X1 = pickle.load(picklein, encoding='latin1')
                    data = X1["data"].reshape(-1, 32, 32, 3)  # not normalized
                    label = X1["labels"]
                if (datasetName == 'Fashion-MNIST'):
                    file = testpath + '/' + datasetName
                    datafmnist = input_data.read_data_sets(file, one_hot=True)
                    data = np.zeros((len(datafmnist.train.images), 28, 28, 1))
                    for i in range(len(datafmnist.train.images)):
                        data[i] = datafmnist.train.images[i].reshape(28, 28, 1)
                    label = datafmnist.train.labels

                maxdatavalue = np.max(data)
                data = data / maxdatavalue
                if (datasetName == 'CIFAR-10'):
                    labelset = np.zeros((len(label), 10))
                    for i in range(len(label)):
                        labelset[i] = np.eye(10)[label[i]]
                if (datasetName == 'Fashion-MNIST'):
                    labelset = label

            mysavednw = graph.get_tensor_by_name('fully_connected_2/BiasAdd:0')
            inputs = graph.get_tensor_by_name('input_x:0')
            prediction = sesst.run(mysavednw, feed_dict={inputs: data})
            obtainedlabel = np.argmax(prediction, axis=1)
            actuallabel = np.argmax(labelset, axis=1)

            print("Test accuracy ::", accuracy_score(actuallabel, obtainedlabel))
            print("Test Macro F1 Score::", f1_score(actuallabel, obtainedlabel, average='macro'))
            print("Test Micro F1 Score::", f1_score(actuallabel, obtainedlabel, average='micro'))

