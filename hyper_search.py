import numpy as np
import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim
import random

from bayes_opt import BayesianOptimization


dataset = np.loadtxt("FM_dataset.dat")

class Net(nn.Module):
	def __init__(self, in_size, out_size,hidden1,hidden2,hidden3):
		super(Net, self).__init__()
		self.a = nn.Linear(in_size,int(hidden1))
		self.d = nn.Linear(int(hidden1),int(hidden2))
		self.e = nn.Linear(int(hidden2),int(hidden3))
		self.f = nn.Linear(int(hidden3), out_size)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				init.xavier_uniform_(m.weight)

	def forward(self, x):
		x = F.relu(self.a(x))
		x = F.relu(self.d(x))
		x = F.relu(self.e(x))
		x = self.f(x)
		return x

def main(hidd1,hidd2,hidd3,learning_rate):

	# check for gpu.
	device = "cpu"
	print(device)
	# create the model.
	dataset = np.loadtxt("FM_dataset.dat")
	#######################################################################
	#                       ** START OF YOUR CODE **
	#######################################################################
	np.random.shuffle(dataset)

	trainRatio = 0.8
	testRatio  = 0.1
	numEpochs  = 500
	batchsize  = 32

	dataX, dataY = dataset[:, 0:3], dataset[:, 3:6]
	# split the dataset
	dataX = torch.stack([torch.Tensor(i) for i in dataX])
	dataY = torch.stack([torch.Tensor(i) for i in dataY])
	# normalise the data
	dataX = F.normalize(dataX)
	dataY = F.normalize(dataY)
	# setup size ratios
	dataSize       = int(dataset.shape[0])
	trainingSize   = int(np.floor(dataSize * trainRatio))
	leftoverSize   = int(dataSize - trainingSize)
	testSize       = int(np.floor(leftoverSize * (testRatio/(1-trainRatio))))
	validationSize = leftoverSize - testSize

	trainX = dataX[:trainingSize]
	valX   = dataX[trainingSize:trainingSize + validationSize]
	testX  = dataX[trainingSize + validationSize:]
	trainY = dataY[:trainingSize]
	valY   = dataY[trainingSize:trainingSize + validationSize]
	testY  = dataY[trainingSize + validationSize:]

	trainloader = utils.DataLoader(
		utils.TensorDataset(trainX, trainY), batch_size=batchsize)
	validloader = utils.DataLoader(
		utils.TensorDataset(valX, valY), batch_size=batchsize)
	testloader = utils.DataLoader(
		utils.TensorDataset(testX, testY), batch_size=batchsize)

	model = Net(3,3,hidd1,hidd2,hidd3).to(device) #3 hyper parameters for sizes

	loss_function = nn.MSELoss()
	optimizer     = optim.Adam(model.parameters(),lr = learning_rate) #1 hyper-parameter

	for epoch in range(1, numEpochs):
		
		train_loss, valid_loss = [], []
		# training part
		model.train()
		for data, target in trainloader:
			data, target = data.to(device), target.to(device)
			optimizer.zero_grad()
			output = model(data)
			loss = loss_function(output, target)
			loss.backward()
			optimizer.step()
			train_loss.append(loss.item())

		# evaluation part
		model.eval()

		for data, target in validloader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			loss = loss_function(output, target)
			valid_loss.append(loss.item())

		print("Epoch:", epoch, "\tTraining Loss: ", round(np.mean(train_loss),4), "\tValid Loss: ", round(np.mean(valid_loss),4))

	# test greatness of model
	model.eval()
	test_loss = 0
	with torch.no_grad():
		for data, target in testloader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			# sum up batch loss
			# for i in range(len(output)):
			#     print(output[i],target[i])
			test_loss += loss_function(output, target).item()

	test_loss /= float(len(testloader.dataset))

	print('\nTest set: Average loss: {:.4f}'.format(test_loss))


	return -test_loss #value to maximize	
	#######################################################################
	#                       ** END OF YOUR CODE **
	#######################################################################


def train_final_model(dataset,optim_params): #train model with optimal parameters 

	dataX, dataY = dataset[:, 0:3], dataset[:, 3:6]

	# split the dataset
	dataX = torch.stack([torch.Tensor(i) for i in dataX])
	dataY = torch.stack([torch.Tensor(i) for i in dataY])

	trainX = F.normalize(dataX)
	trainY = F.normalize(dataY)

	numEpochs  = 500
	batchsize  = 32

	device = "cpu"

	trainloader = utils.DataLoader(utils.TensorDataset(trainX, trainY), batch_size=batchsize)

	np.random.shuffle(dataset)

	model = Net(3,3,optim_params['hidd1'],optim_params['hidd2'],optim_params['hidd3']).to(device) #3 hyper parameters for sizes

	loss_function = nn.MSELoss()
	optimizer     = optim.Adam(model.parameters(),lr = optim_params['learning_rate']) #1 hyper-parameter

	for epoch in range(1, numEpochs):

		train_loss, valid_loss = [], []
		# training part
		model.train()
		for data, target in trainloader:
			data, target = data.to(device), target.to(device)

			optimizer.zero_grad()
			output = model(data)
			loss = loss_function(output, target)
			loss.backward()
			optimizer.step()
			train_loss.append(loss.item())

	torch.save(model, './Model/optim_model.pt')

	return(model)
	# test greatness of model

def test_final_model(dataset):

	dataX, dataY = dataset[:, 0:3], dataset[:, 3:6]

	# split the dataset
	dataX = torch.stack([torch.Tensor(i) for i in dataX])
	dataY = torch.stack([torch.Tensor(i) for i in dataY])

	testX = F.normalize(dataX) #normalization of data
	testY = F.normalize(dataY)

	device = "cpu" #set device 
	batchsize  = 32 #set batchsize

	testloader = utils.DataLoader(utils.TensorDataset(testX, testY), batch_size=batchsize)

	np.random.shuffle(dataset) #shuffle data

	model = torch.load('./Model/optim_model.pt') #load model previously trained
	loss_function = nn.MSELoss()

	# test greatness of model
	model.eval()
	test_loss = 0
	with torch.no_grad():
		for data, target in testloader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			# sum up batch loss
			# for i in range(len(output)):
			#     print(output[i],target[i])
			test_loss += loss_function(output, target).item()

	test_loss /= float(len(testloader.dataset))

	print('\nTest set: Average loss: {:.4f}'.format(test_loss))

	return(True)


if __name__ == "__main__":

	random.seed(42)

	####	UNCOMMENT PART BELOW TO PROCESS BAYESIAN OPTIMISATION
	# pbounds = {'hidd1': (3, 30), 'hidd2': (3, 30), 'hidd3': (3, 30), 'learning_rate': (0.0001,0.01)}
	# optimizer = BayesianOptimization(f=main,pbounds=pbounds,random_state=1)
	# print("######### STARTING BAYESIAN OPTIMIZATION #######\n\n\n")
	# optimizer.maximize(init_points=4,n_iter=50)
	# optim_params = optimizer.max
	# print("RESULTS :", optim_params)
	# print("\n\n\n######### END OF OPTIMIZATION ########")
	####

	####UNCOMMENT PART BELOW TO USE COMPUTED OPTIMAL PARAMETERS
	# optim_params = {'hidd1':30, 'hidd2':30,'hidd3':30,'learning_rate':0.008719}
	# final_model = train_final_model(dataset,optim_params)
	
	# print(final_model)
	####

	####TEST IF SAVED MODEL IS WORKING (COMMENT IF OPTIMISING OR TRAINING FINAL VERSION)
	result = test_final_model(dataset)
	####


##### BEST RESULT OBTAINED FOR :
##
## hidden layer 1 : 30 
##
## hidden layer 2 : 30
##
## hidden layer 3 : 30
##
## learning rate : 0.0008719
##
## =====> Test loss : 0.000605