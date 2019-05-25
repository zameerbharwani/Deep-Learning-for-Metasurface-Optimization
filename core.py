#!/usr/bin/python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Dense, LeakyReLU


def data_prep():
	"""
	Prepare the train,test, and validation data
	"""

	input_data = np.genfromtxt('Input.csv', delimiter=',', skip_header=1)
	output_data = np.transpose(np.genfromtxt(base + 'Output.csv', delimiter=',', skip_header=0))

	mean = input_data.mean(axis = 0) # mean value for each feature
	variance = input_data.std(axis = 0)**2 # variance for each feature
	normalized_input = tf.nn.batch_normalization(input_data,mean,variance,None,None,1e-9,None) 
	# normalize the input data so that all input features have the same range and same influence on the regression

	train_x, test_x, train_y, test_y = train_test_split(input_data,output_data,test_size=float(0.2)) # split initial data into train/test 70/30%
	test_x,val_x,test_y,val_y = train_test_split(test_x,test_y,test_size=float(0.5)) # split the test data 50/50 into test and validation

	np.savetxt('train_x.txt',train_x)
	np.savetxt('train_y.txt',train_y)
	np.savetxt('test_x.txt',test_x)
	np.savetxt('test_y.txt',test_y)
	np.savetxt('val_x.txt',val_x)
	np.savetxt('val_y.txt',val_y)

	return train_x, test_x, train_y, test_y, val_x,val_y

def load_data():

	train_x = np.genfromtxt("train_x.txt")
	train_y = np.genfromtxt("train_y.txt")
	test_x = np.genfromtxt("test_x.txt")
	test_y = np.genfromtxt("test_y.txt")
	val_x = np.genfromtxt("val_x.txt")
	val_y = np.genfromtxt("val_y.txt")

	return train_x, test_x, train_y, test_y, val_x, val_y


def network(train_x, test_x, train_y, test_y,val_x,val_y,reuse_weights,alpha,epochs,batch_size,neurons,l2_w,l2_b,fname,load_model_,optimizer,graph):

	""" 
	Architecture: 5 input parameters (L,W,H,Ux,Uy), 3 hidden layers
	(Leaky) ReLU activation for hidden layers, final layer has identity activcation since output can take on negative values
	Output size 71, each point represneting the phase at a wavelength starting at 450 nm in increments of 5 nm up to 800 nm  

	Note: If you think you might need more than 3-5 layers, write a loop to build your own custom layers to avoid a thick codebase

	"""

	if (load_model_ == 'True' and os.path.isfile(os.getcwd()+'/my_model.h5')):

		print ("==== Loading Model ====")

		model = load_model('my_model.h5')

	elif (reuse_weights == 'True' and os.path.isfile(os.getcwd()+'/'+ fname)): 

		print ("==== Re-using loaded weights ====")

		model = Sequential()

		model.add(Dense(neurons, input_dim = 5, use_bias = True ,kernel_regularizer = regularizers.l2(l2_w), bias_regularizer = regularizers.l2(l2_b)))
		model.add(LeakyReLU(alpha = alpha))

		model.add(Dense(neurons, use_bias = True,kernel_regularizer = regularizers.l2(l2_w), bias_regularizer = regularizers.l2(l2_b), ))
		model.add(LeakyReLU(alpha = alpha))

		model.add(Dense(neurons, use_bias = True,kernel_regularizer = regularizers.l2(l2_w), bias_regularizer = regularizers.l2(l2_b)))
		model.add(LeakyReLU(alpha = alpha))

		model.add(Dense(neurons, use_bias = True, kernel_regularizer = regularizers.l2(l2_w), bias_regularizer = regularizers.l2(l2_b)))
		model.add(LeakyReLU(alpha = alpha))

		model.add(Dense(71, activation = 'linear', use_bias = True, kernel_regularizer = regularizers.l2(l2_w), bias_regularizer = regularizers.l2(l2_b)))

		model.load_weights(fname)


	else: # start from scratch

		print ("==== Starting from Scratch ====")


		model = Sequential()

		model.add(Dense(neurons, input_dim = 5, use_bias =True, kernel_initializer = 'glorot_normal', bias_initializer = 'glorot_normal',kernel_regularizer=regularizers.l2(l2_w), bias_regularizer=regularizers.l2(l2_b)))
		model.add(LeakyReLU(alpha=alpha))

		model.add(Dense(neurons, use_bias =True, kernel_initializer = 'glorot_normal', bias_initializer = 'glorot_normal',kernel_regularizer=regularizers.l2(l2_w), bias_regularizer=regularizers.l2(l2_b)))
		model.add(LeakyReLU(alpha=alpha))

		model.add(Dense(neurons, use_bias =True, kernel_initializer = 'glorot_normal', bias_initializer = 'glorot_normal',kernel_regularizer=regularizers.l2(l2_w), bias_regularizer=regularizers.l2(l2_b)))
		model.add(LeakyReLU(alpha=alpha))

		model.add(Dense(neurons, use_bias =True, kernel_initializer = 'glorot_normal', bias_initializer = 'glorot_normal',kernel_regularizer=regularizers.l2(l2_w), bias_regularizer=regularizers.l2(l2_b)))
		model.add(LeakyReLU(alpha=alpha))

		model.add(Dense(71, activation = 'linear', use_bias = True, kernel_initializer = 'glorot_normal', bias_initializer = 'glorot_normal',kernel_regularizer=regularizers.l2(l2_w), bias_regularizer=regularizers.l2(l2_b)))
	
	model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

	history = model.fit(train_x,train_y, epochs=epochs, batch_size=batch_size,  verbose=2, validation_data=(val_x, val_y))

	score = model.evaluate(test_x,test_y, batch_size = batch_size,verbose = 1) # test network

	print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

	save_weights = True if raw_input("Do you want to save the weights? (y/n) ") == 'y' else False
	save_model = True if raw_input("Do you want to save the model? (y/n) ") == 'y' else False

	if save_weights:  # save the weights
		model.save('my_model_weights.h5')

	if save_model: # save entire model
		model.save('my_model.h5')

	print  model.summary()

	if graph == 'True':
		gen_graphs(test_x,test_y,val_x,val_y,model)

	return score

def gen_graphs(test_x,test_y,val_x,val_y,model):

	if not (os.path.isdir(os.path.join(os.getcwd(), 'test_results'))):
		os.mkdir('test_results')

	if not (os.path.isdir(os.path.join(os.getcwd(), 'validation_results'))):
		os.mkdir('validation_results')

	lambda_WL = np.arange(450,805,5)

	for i in range(len(test_x)):
		prediction = model.predict(test_x[i:i+1])
		fig = plt.figure()
		plt.plot(lambda_WL,prediction.reshape(len(lambda_WL),),label = "Prediction")
		plt.plot(lambda_WL,test_y[i], label= "Actual")
		plt.title("L=%s nm, W=%s nm , H=%s nm, Ux=%s nm, Uy=%s nm"%(test_x[i][0],test_x[i][1],test_x[i][2],test_x[i][3],test_x[i][4]))
		plt.xlabel("Wavelength (nm)")
		plt.ylabel("Phase (rad)")
		plt.legend(loc='best')
		plt.savefig(os.getcwd()+'/test_results/test_%s.png'%(str(i)))
		plt.close()

	for i in range(len(val_x)):
		prediction = model.predict(val_x[i:i+1])
		fig = plt.figure()
		plt.title("Legend Values: L/W/H/Ux/Uy")
		plt.plot(lambda_WL,prediction.reshape(len(lambda_WL),),label = "Prediction")
		plt.plot(lambda_WL,val_y[i], label= "Actual")
		plt.title("L=%s nm, W=%s nm , H=%s nm, Ux=%s nm, Uy=%s nm"%(val_x[i][0],val_x[i][1],val_x[i][2],val_x[i][3],val_x[i][4]))
		plt.xlabel("wavelength (nm)")
		plt.ylabel("Phase (rad)")
		plt.legend(str(val_x[i:i+1]),loc='best')
		plt.savefig(os.getcwd()+'/validation_results/val_%s.png'%(str(i)))
		plt.close()	


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--load_data', default = 'True', help ="whether to load the test/validate/train split data; false to resplit it again")
	parser.add_argument('--reuse_weights', default = 'True', help ="reuse saved weights")
	parser.add_argument('--alpha', default = 0.05, help ="LeakyReLU parameter", type = float)
	parser.add_argument('--l2_w', default = 0.015, help ="regularization parameter for weights", type = float)
	parser.add_argument('--l2_b', default = 0.015, help ="regularization parameter for biases", type = float)
	parser.add_argument('--epochs', default = 100, help ="number of epochs", type = int)
	parser.add_argument('--batch_size', default = 15, help ="batch size", type = int)
	parser.add_argument('--neurons', default = 200, help ="number of neurons", type = int)
	parser.add_argument('--fname', default = "my_model_weights.h5", help = "saved weights file name", type = str)
	parser.add_argument('--load_model_', default = 'False', help = "load model")
	parser.add_argument('--optimizer', default = "adam", help = "optimizer", type = str)
	parser.add_argument('--graph', default = "False", help = "whether to plot and save graphs")
	parser.parse_args()
	args = parser.parse_args()

	if args.load_data == 'False':

		train_x, test_x, train_y, test_y,val_x,val_y = data_prep()

	else:

		train_x, test_x, train_y, test_y,val_x,val_y = load_data()

	score = network(train_x, test_x, train_y, test_y,val_x,val_y,args.reuse_weights,args.alpha,args.epochs,args.batch_size,args.neurons,args.l2_w,args.l2_b,args.fname,args.load_model_,args.optimizer,args.graph)
	
	print (score)