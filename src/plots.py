import matplotlib.pyplot as mpl

########### CIFAR #############
################### TASK 1#######################

'''
x = ['[64 64]','[64 64 64]','[64 64 64 64]','[64 64 64 64 64]']
y = [0.66, 0.823,0.477,0.107]
mpl.plot(x,y)
mpl.title('accuracy_CIFAR_task1')
mpl.show()
mpl.savefig('accuracy_CIFAR_task1.png', format='png', dpi=500)
mpl.close()

x = ['[64 64]','[64 64 64]','[64 64 64 64]','[64 64 64 64 64]']
y = [0.58035,0.79089,0.37996,0.0193]
mpl.plot(x,y)
mpl.title('F1score_macro_CIFAR_task1')
mpl.show()
mpl.savefig('trainF1score_macro_CIFAR_task1')
mpl.close()

x = ['[64 64]','[64 64 64]','[64 64 64 64]','[64 64 64 64 64]']
y = [0.66, 0.823,0.477,0.107]
mpl.plot(x,y)
mpl.title('F1score_micro_CIFAR_task1')
mpl.show()
mpl.savefig('F1score_micro_CIFAR_task1')
mpl.close()

################## TASK 2 #########################
x = ['[32 64 128]','[64 64 64]','[32 32 32]','[128 64 32]']
y = [0.775,0.823,0.657,0.916]
mpl.plot(x,y)
mpl.title('accuracy_CIFAR_task2')
mpl.show()
mpl.savefig('accuracy_CIFAR_task2.png', format='png', dpi=500)
mpl.close()

x = ['[32 64 128]','[64 64 64]','[32 32 32]','[128 64 32]']
y = [0.690458,0.79089,0.594,0.9160035]
mpl.plot(x,y)
mpl.title('F1score_macro_CIFAR_task2')
mpl.show()
mpl.savefig('trainF1score_macro_CIFAR_task2')
mpl.close()

x = ['[32 64 128]','[64 64 64]','[32 32 32]','[128 64 32]']
y = [0.775,0.823,0.657,0.916]
mpl.plot(x,y)
mpl.title('F1score_micro_CIFAR_task2')
mpl.show()
mpl.savefig('F1score_micro_CIFAR_task2')
mpl.close()

################# TASK 3 ########################
############# [64 64] ##########################
x = ['sigmoid','tanh','relu','swish']
y = [0.098,0.963,0.849,0.852]
mpl.plot(x,y)
mpl.title('accuracy_CIFAR_task3: [64 64]')
mpl.show()
mpl.savefig('accuracy_CIFAR_task3_1.png', format='png', dpi=500)
mpl.close()

x = ['sigmoid','tanh','relu','swish']
y = [0.0178,0.9639,0.820034,0.82147]
mpl.plot(x,y)
mpl.title('F1score_macro_CIFAR_task3: [64 64]')
mpl.show()
mpl.savefig('trainF1score_macro_CIFAR_task3_1')
mpl.close()

x = ['sigmoid','tanh','relu','swish']
y = [0.098,0.963,0.849,0.852]
mpl.plot(x,y)
mpl.title('F1score_micro_CIFAR_task3:[64 64]')
mpl.show()
mpl.savefig('F1score_micro_CIFAR_task3_1')
mpl.close()

#################### TASK 3 ######################
################### [64 64 64] ###################
x = ['sigmoid','tanh','relu','swish']
y = [0.097,0.827,0.929,0.84]
mpl.plot(x,y)
mpl.title('accuracy_CIFAR_task3: [64 64 64]')
mpl.show()
mpl.savefig('accuracy_CIFAR_task3_2.png', format='png', dpi=500)
mpl.close()

x = ['sigmoid','tanh','relu','swish']
y = [0.01768,0.79488,0.928146,0.8043]
mpl.plot(x,y)
mpl.title('F1score_macro_CIFAR_task3: [64 64 64]')
mpl.show()
mpl.savefig('trainF1score_macro_CIFAR_task3_2')
mpl.close()

x = ['sigmoid','tanh','relu','swish']
y = [0.097,0.827,0.929,0.84]
mpl.plot(x,y)
mpl.title('F1score_micro_CIFAR_task3:[64 64 64]')
mpl.show()
mpl.savefig('F1score_micro_CIFAR_task3_2')
mpl.close()

################### TASK 4 #############################
x = ['xavier','stddev']
y = [0.66,0.98]
mpl.plot(x,y)
mpl.title('accuracy_CIFAR_task4: [64 64], relu')
mpl.show()
mpl.savefig('accuracy_CIFAR_task4.png', format='png', dpi=500)
mpl.close()

x = ['xavier','stddev']
y = [0.58035, 0.98008]
mpl.plot(x,y)
mpl.title('F1score_macro_CIFAR_task4: [64 64], relu')
mpl.show()
mpl.savefig('F1score_macro_CIFAR_task4')
mpl.close()

x = ['xavier','stddev']
y = [0.66,0.98]
mpl.plot(x,y)
mpl.title('F1score_micro_CIFAR_task4:[64 64], relu')
mpl.show()
mpl.savefig('F1score_micro_CIFAR_task4')
mpl.close()
################### Fashion MNIST ########################
################### TASK 1#######################


x = ['[64 64 64]','[64 64 64 64]','[64 64 64 64 64]', '[64 64 64 64 64 64]', '[64 64 64 64 64 64 64]']
y = [0.6565, 0.9089,0.74073,0.9036,0.80818]
mpl.plot(x,y)
mpl.title('accuracy_FashionMNIST_task1')
mpl.show()
mpl.savefig('accuracy_FashionMNIST_task1.png', format='png', dpi=500)
mpl.close()

x = ['[64 64 64]','[64 64 64 64]','[64 64 64 64 64]', '[64 64 64 64 64 64]', '[64 64 64 64 64 64 64]']
y = [0.5965,0.9085,0.6929,0.9050,0.78043]
mpl.plot(x,y)
mpl.title('F1score_macro_FashionMNIST_task1')
mpl.show()
mpl.savefig('F1score_macro_FashionMNIST_task1')
mpl.close()

x = ['[64 64 64]','[64 64 64 64]','[64 64 64 64 64]', '[64 64 64 64 64 64]', '[64 64 64 64 64 64 64]']
y = [0.6565, 0.9089,0.74073,0.9036,0.80818]
mpl.plot(x,y)
mpl.title('F1score_micro_FashionMNIST_task1')
mpl.show()
mpl.savefig('F1score_micro_FashionMNIST_task1')
mpl.close()

################## TASK 2 #########################
x = ['[64 64 64 64]','[64 128 64 128]','[64 128 256 512]','[64 128 128 64]','[32,64,128,32]']
y = [0.9089,0.57309,0.6321818,0.915818,0.638181]
mpl.plot(x,y)
mpl.title('accuracy_FashionMNIST_task2')
mpl.show()
mpl.savefig('accuracy_FashionMNIST_task2.png', format='png', dpi=500)
mpl.close()

x = ['[64 64 64 64]','[64 128 64 128]','[64 128 256 512]','[64 128 128 64]','[32,64,128,32]']
y = [0.9085,0.4951,0.581294,0.915653,0.581695]
mpl.plot(x,y)
mpl.title('F1score_macro_FashionMNIST_task2')
mpl.show()
mpl.savefig('F1score_macro_FashionMNIST_task2')
mpl.close()

x = ['[64 64 64 64]','[64 128 64 128]','[64 128 256 512]','[64 128 128 64]','[32,64,128,32]']
y = [0.9089,0.57309,0.6321818,0.915818,0.638181]
mpl.plot(x,y)
mpl.title('F1score_micro_FashionMNIST_task2')
mpl.show()
mpl.savefig('F1score_micro_FashionMNIST_task2')
mpl.close()

################# TASK 3 ########################
############# [64 64 64 64] ##########################
x = ['sigmoid','tanh','relu','swish']
y = [0.102545,0.90909,0.813818,0.828]
mpl.plot(x,y)
mpl.title('accuracy_FashionMNIST_task3: [64 64 64 64]')
mpl.show()
mpl.savefig('accuracy_FashionMNIST_task3_1.png', format='png', dpi=500)
mpl.close()

x = ['sigmoid','tanh','relu','swish']
y = [0.186015,0.90883,0.7829,0.79535]
mpl.plot(x,y)
mpl.title('F1score_macro_FashionMNIST_task3: [64 64 64 64]')
mpl.show()
mpl.savefig('F1score_macro_FashionMNIST_task3_1')
mpl.close()

x = ['sigmoid','tanh','relu','swish']
y = [0.102545,0.90909,0.813818,0.828]
mpl.plot(x,y)
mpl.title('F1score_micro_FashionMNIST_task3:[64 64 64 64]')
mpl.show()
mpl.savefig('F1score_micro_FashionMNIST_task3_1')
mpl.close()

#################### TASK 3 ######################
################### [64 64 64 64 64] ###################
x = ['sigmoid','tanh','relu','swish']
y = [0.09909,0.8256,0.75218,0.912727]
mpl.plot(x,y)
mpl.title('accuracy_FashionMNIST_task3: [64 64 64 64 64]')
mpl.show()
mpl.savefig('accuracy_FashionMNIST_task3_2.png', format='png', dpi=500)
mpl.close()

x = ['sigmoid','tanh','relu','swish']
y = [0.018031,0.7949,0.7005,0.913579]
mpl.plot(x,y)
mpl.title('F1score_macro_FashionMNIST_task3: [64 64 64 64 64]')
mpl.show()
mpl.savefig('F1score_macro_FashionMNIST_task3_2')
mpl.close()

x = ['sigmoid','tanh','relu','swish']
y = [0.09909,0.8256,0.75218,0.912727]
mpl.plot(x,y)
mpl.title('F1score_micro_FashionMNIST_task3:[64 64 64 64 64]')
mpl.show()
mpl.savefig('F1score_micro_FashionMNIST_task3_2')
mpl.close()

#################### TASK 3 ######################
################### [64 64 64 64 64 64] ###################
x = ['sigmoid','tanh','relu','swish']
y = [0.091636,0.89818,0.805818,0.9094545]
mpl.plot(x,y)
mpl.title('accuracy_FashionMNIST_task3: [64 64 64 64 64 64]')
mpl.show()
mpl.savefig('accuracy_FashionMNIST_task3_3.png', format='png', dpi=500)
mpl.close()

x = ['sigmoid','tanh','relu','swish']
y = [0.0167888,0.8998578,0.777897,0.9084237]
mpl.plot(x,y)
mpl.title('F1score_macro_FashionMNIST_task3: [64 64 64 64 64 64]')
mpl.show()
mpl.savefig('F1score_macro_FashionMNIST_task3_3')
mpl.close()

x = ['sigmoid','tanh','relu','swish']
y = [0.091636,0.89818,0.805818,0.9094545]
mpl.plot(x,y)
mpl.title('F1score_micro_FashionMNIST_task3:[64 64 64 64 64 64]')
mpl.show()
mpl.savefig('F1score_micro_FashionMNIST_task3_3')
mpl.close()

################### TASK 4 #############################
x = ['xavier','stddev']
y = [0.80764,0.4885]
mpl.plot(x,y)
mpl.title('accuracy_FashionMNIST_task4: [64 128 256 512 128 256], relu')
mpl.show()
mpl.savefig('accuracy_FashionMNIST_task4.png', format='png', dpi=500)
mpl.close()

x = ['xavier','stddev']
y = [0.77964, 0.4166]
mpl.plot(x,y)
mpl.title('F1score_macro_FashionMNIST_task4: [64 128 256 512 128 256], relu')
mpl.show()
mpl.savefig('F1score_macro_FashionMNIST_task4')
mpl.close()

x = ['xavier','stddev']
y = [0.80764,0.4885]
mpl.plot(x,y)
mpl.title('F1score_micro_FashionMNIST_task4:[64 128 256 512 128 256], relu')
mpl.show()
mpl.savefig('F1score_micro_FashionMNIST_task4')
mpl.close()
'''
'''
################# TASK 5 ###################
x = [0.1,0.2,0.3,0.4,0.5]
y = [0.15833,0.253125,0.249285,0.223169,0.204]
mpl.plot(x,y)
mpl.title('Clusteringaccuracy_CIFAR_task5: [64 64], tanh')
mpl.xlabel('Train set percentage')
mpl.ylabel('Clustering Accuracy CIFAR-10')
mpl.show()
mpl.savefig('Clusteraccuracy_CIFAR_task5.png', format='png', dpi=500)
mpl.close()

x = [0.1,0.2,0.3,0.4,0.5]
y = [0.2661,0.212818,0.27961,0.176,0.185745]
mpl.plot(x,y)
mpl.title('Clusteringaccuracy_FashionMNIST_task5: [64 64 64 64], tanh')
mpl.xlabel('Train set percentage')
mpl.ylabel('Clustering Accuracy Fashion MNIST')
mpl.show()
mpl.savefig('Clusteraccuracy_FashionMNIST_task5.png', format='png', dpi=500)
mpl.close()
'''
'''
############# CIFAR ################
############# TASK 1 with Filter size ###########
x = [3,4,5,6,7]
y = [0.395,0.249,0.262,0.266,0.473]
mpl.plot(x,y)
mpl.title('accuracy_CIFAR_task1a')
mpl.show()
mpl.savefig('accuracy_CIFAR_task1a.png', format='png', dpi=500)
mpl.close()

x = [3,4,5,6,7]
y = [0.33247,0.18228,0.19779,0.2159,0.44637]
mpl.plot(x,y)
mpl.title('F1score_macro_CIFAR_task1a')
mpl.show()
mpl.savefig('trainF1score_macro_CIFAR_task1a')
mpl.close()

x = [3,4,5,6,7]
y = [0.395,0.249,0.262,0.266,0.473]
mpl.plot(x,y)
mpl.title('F1score_micro_CIFAR_task1a')
mpl.show()
mpl.savefig('F1score_micro_CIFAR_task1a')
mpl.close()
################## TASK 2 ##############
x = ['[2,2]','[3,3]','[4,4]','[5,5]']
y = [0.972,0.97,0.954,0.948]
mpl.plot(x,y)
mpl.title('accuracy_CIFAR_task2a')
mpl.show()
mpl.savefig('accuracy_CIFAR_task2a.png', format='png', dpi=500)
mpl.close()

x = ['[2,2]','[3,3]','[4,4]','[5,5]']
y = [0.97168,0.969151,0.95393,0.9479945]
mpl.plot(x,y)
mpl.title('F1score_macro_CIFAR_task2a')
mpl.show()
mpl.savefig('trainF1score_macro_CIFAR_task2a')
mpl.close()

x = ['[2,2]','[3,3]','[4,4]','[5,5]']
y = [0.972,0.97,0.954,0.948]
mpl.plot(x,y)
mpl.title('F1score_micro_CIFAR_task2a')
mpl.show()
mpl.savefig('F1score_micro_CIFAR_task2a')
mpl.close()
'''
################## Task 1 Fashion MNIST ###########
x = [3,4,5,6]
y = [0.46,0.65818,0.642727,0.807818]
mpl.plot(x,y)
mpl.title('accuracy_FashionMNIST_task1a')
mpl.show()
mpl.savefig('accuracy_FashionMNIST_task1a.png', format='png', dpi=500)
mpl.close()

x = [3,4,5,6]
y = [0.38651,0.5905,0.58167,0.77531376]
mpl.plot(x,y)
mpl.title('F1score_macro_FashionMNIST_task1a')
mpl.show()
mpl.savefig('F1score_macro_FashionMNIST_task1a')
mpl.close()

x = [3,4,5,6]
y = [0.46,0.65818,0.642727,0.807818]
mpl.plot(x,y)
mpl.title('F1score_micro_FashionMNIST_task1a')
mpl.show()
mpl.savefig('F1score_micro_FashionMNIST_task1a')
mpl.close()

################## TASK 2 #########################
x = [1,2,3,4]
y = [0.606545,0.61527,0.61709,0.5783636]
mpl.plot(x,y)
mpl.xlabel('num')
mpl.ylabel("accuracy")
mpl.title('accuracy_FashionMNIST_task2a; all ones multiplied by num; 6 layer')
mpl.show()
mpl.savefig('accuracy_FashionMNIST_task2a.png', format='png', dpi=500)
mpl.close()

x = [1,2,3,4]
y = [0.557176,0.56346,0.515008,0.54999]
mpl.plot(x,y)
mpl.xlabel('num')
mpl.ylabel("F1 Score Macro")
mpl.title('F1score_macro_FashionMNIST_task2a; all ones multiplied by num; 6 layer')
mpl.show()
mpl.savefig('F1score_macro_FashionMNIST_task2a')
mpl.close()

x = [1,2,3,4]
y = [0.606545,0.61527,0.61709,0.5783636]
mpl.plot(x,y)
mpl.xlabel('num')
mpl.ylabel("F1 Score Micro")
mpl.title('F1score_micro_FashionMNIST_task2a; all ones multiplied by num; 6 layer')
mpl.show()
mpl.savefig('F1score_micro_FashionMNIST_task2a')
mpl.close()