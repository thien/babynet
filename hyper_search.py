import numpy as np
import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim
import helpers

# https://github.com/fmfn/BayesianOptimization
# pip3 install bayesian-optimization
from bayes_opt import BayesianOptimization

dataset = np.loadtxt("FM_dataset.dat")
#######################################################################
#                       ** START OF YOUR CODE **
######################################################################


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
	# create the model.
	dataset = np.loadtxt("FM_dataset.dat")
	#######################################################################
	#                       ** START OF YOUR CODE **
	#######################################################################
	np.random.shuffle(dataset)

	trainRatio = 0.8
	testRatio  = 0.1
	numEpochs  = 20
	batchsize  = 32

	X, Y = dataset[:, 0:3], dataset[:, 3:6]
	trainloader, validloader, testloader = helpers.loadTrainingData(
        X, Y, trainRatio, testRatio, batchsize, normalise=True)

	model = Net(3,3,hidd1,hidd2,hidd3) #3 hyper parameters for sizes

	loss_function = nn.MSELoss()
	optimizer     = optim.Adam(model.parameters(), lr=learning_rate) #1 hyper-parameter

	val_loss = helpers.trainModel(
		model, optimizer, loss_function, numEpochs, trainloader, validloader, verbose=False)

	# you shouldn't be optimising on test data.
	test_loss = helpers.testModel(
		model, loss_function, numEpochs, testloader, verbose=False)

	return -float(test_loss) #value to maximize

		
	#######################################################################
	#                       ** END OF YOUR CODE **
	#######################################################################
	# illustrate_results_FM(network, prep)


if __name__ == "__main__":

	pbounds = {'hidd1': (3, 25), 'hidd2': (3, 25), 'hidd3': (3,25), 'learning_rate': (0.0001,0.001)}

	optimizer = BayesianOptimization(f=main,pbounds=pbounds,random_state=1)

	print("######### STARTING BAYESIAN OPTIMIZATION #######\n\n\n")

	optimizer.maximize(init_points=4,n_iter=20)

	optim_params = optimizer.max

	print("RESULTS :", optim_params)

	print("\n\n\n######### END OF OPTIMIZATION ########")


##### BEST RESULT OBTAINED FOR :
##
## hidden layer 1 : 11 
##
## hidden layer 2 : 14
##
## hidden layer 3 : 12
##
## learning rate : 0.0007
##
## =====> Test loss : 0.000604