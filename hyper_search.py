import numpy as np
import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim
import random
import helpers
from nn_lib import Preprocessor
#from illustrate import illustrate_results_FM
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from bayes_opt import BayesianOptimization


seed = 1337
np.random.seed(seed)
torch.manual_seed(seed)

defaults = {
    'l1':20, 
    'l2':15, 
    'l3':10, 
    'l4':7,
    'lr':1e-3,
    'save':False,
    'epochs':20,
    'filepath': './Model/fm_model.pt'
}

class Net(nn.Module):
    def __init__(self, in_size, out_size, hidd1, hidd2, hidd3, hidd4):
        super(Net, self).__init__()
        self.a = nn.Linear(in_size,hidd1)
        self.d = nn.Linear(hidd1,hidd2)
        self.e = nn.Linear(hidd2,hidd3)
        self.g = nn.Linear(hidd3,hidd4)
        self.f = nn.Linear(hidd4, out_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = F.relu(self.a(x))
        x = F.relu(self.d(x))
        x = F.relu(self.e(x))
        x = F.relu(self.g(x))
        # we do not have an activation
        # function on the last layer deliberately.
        x = self.f(x)
        return x

def blackbox(l1,l2,l3,l4,lr):
    """
    Compressed runthrough of training the model.
    This is used in the BayesianOptimisation function
    in order to deduce the best hyperparameters during
    its search. You shouldn't use this to call the
    model. Use run() instead.
    """
    # silly optimiser will make floats of these.
    l1,l2,l3,l4 = int(l1),int(l2),int(l3),int(l4)
    # load dataset file
    dataset = np.loadtxt("FM_dataset.dat")
    np.random.shuffle(dataset)
    # setup base parameters
    trainRatio = 0.8
    testRatio = 0.1
    numEpochs = 20
    batchsize = 64
    X, Y = dataset[:, 0:3], dataset[:, 3:]
    xDim, yDim = X.shape[-1], Y.shape[-1]
    trainloader, validloader, _ = helpers.loadTrainingData(
        X, Y, trainRatio, testRatio, batchsize)
    # initiate the model
    model = Net(xDim, yDim, l1, l2, l3, l4)
    # set up hyperparameters
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # train the model
    return -helpers.trainModel(
        model, optimizer, loss_function, 
        numEpochs, trainloader, validloader, 
        verbose=False)

def optimise():
    print("Initiating Bayesian Optimisation...")
    # initiate parameters
    params = {
        'l1': (3, 300), 
        'l2': (3, 300), 
        'l3': (3, 300),
        'l4': (3, 300),
        'lr': (0.0001, 0.001)
    }

    # initiate optimiser
    bayes = BayesianOptimization(
                f=blackbox,
                pbounds=params,
                random_state=seed)

    # search hyperparameter space
    bayes.maximize(init_points=4,n_iter=100)
    print("Finished bayesian optimisation.")
    # retrieve the best hyperparameter
    best = bayes.max

    # convert layers variables to ints
    toIntVariableNames = ['l1', 'l2', 'l3', 'l4']
    for layer in toIntVariableNames:
        best['params'][layer] = int(best['params'][layer])

    # initiate parameters
    arguments = copy.deepcopy(defaults)
    for key in best:
        arguments[key] = best[key]
    arguments['save'] = True
    arguments['epochs'] = 50

    print("Training Model with found parameters.. ", end="")
    # train the bot with the optimal parameters
    # and save it.
    run(verbose=False, showBot=False, args=arguments)
    print("done.")


def run(verbose=True, showBot=False, args=defaults):
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    np.random.shuffle(dataset)
    
    trainRatio = 0.8
    testRatio = 0.1
    numEpochs = args['epochs']
    batchsize = 64

    # split dataset
    X, Y = dataset[:, 0:3], dataset[:, 3:]

    # get input and output dimensions
    xDim, yDim = X.shape[-1], Y.shape[-1]

    # create dataloaders
    trainloader, validloader, testloader = helpers.loadTrainingData(
        X, Y, trainRatio, testRatio, batchsize)

    # initiate the model
    model = Net(
        xDim, yDim, args['l1'], args['l2'], args['l3'], args['l4'])

    # set up hyperparameters
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    # train the model
    val_loss = helpers.trainModel(model, optimizer, loss_function, numEpochs, trainloader, validloader, verbose=verbose)

    if args['save']:
        torch.save(model, args['filepath'])

    # test the model
    helpers.testModel(model, loss_function, numEpochs, testloader, checkAccuracy=True, verbose=verbose)

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    if showBot:
        illustrate_results_ROI(model)

    return val_loss


if __name__ == "__main__":

	optimise()


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