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

import copy 

seed = 1337
np.random.seed(seed)
torch.manual_seed(seed)

defaults = {
    'l1':128, 
    'l2':64, 
    'l3':32, 
    'l4':8,
    'lr':0.001,
    'save':False,
    'epochs':50,
    'filepath': 'best_model_reg.pth'
}

# create the model.
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
    
def main(args = defaults):
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    np.random.shuffle(dataset)
    
    trainRatio = 0.8
    testRatio  = 0.1
    numEpochs  = 30
    batchsize  = 64

    # shuffle the dataset prior
    np.random.shuffle(dataset)

    # preprocess data
    prep = Preprocessor(dataset)
    dataset = prep.apply(dataset)

    # retrieve X and Y columns
    X, Y = dataset[:, 0:3], dataset[:, 3:6]

    # create loaders
    trainloader, validloader, testloader = helpers.loadTrainingData(
        X, Y, trainRatio, testRatio, batchsize)

    # initiate the model
    model = Net(3,3,args['l1'], args['l2'], args['l3'], args['l4'])

    # load hyperparameters
    loss_function = nn.MSELoss()
    optimizer     = optim.Adam(model.parameters())

    # train the model
    helpers.trainModel(model, optimizer, loss_function, numEpochs, trainloader, validloader)
    # evaluate the model
    helpers.testModel(model, loss_function, numEpochs, testloader)
    
    torch.save(model.state_dict(), 'best_model_reg.pth')
    
    x_store, y_store = [],[]
    i =0
    for x,y in testloader:
        if i < 1: 
            x_store = np.array(x)
            y_store = np.array(y)
        else:
            np.concatenate((x_store, np.array(x)), axis = 0)
            np.concatenate((y_store, np.array(y)), axis = 0)
        i = 1

    evaluate_architecture(model, torch.Tensor(x_store), torch.Tensor(y_store), prep)
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # data is normalised in this function
    illustrate_results_FM(model, prep)

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
    numEpochs = 30
    batchsize = 64
    X, Y = dataset[:, 0:3], dataset[:, 3:]
    xDim, yDim = X.shape[-1], Y.shape[-1]
    trainloader, validloader, _ = helpers.loadTrainingData(
        X, Y, trainRatio, testRatio, batchsize)
    # initiate the model
    model = Net(xDim, yDim, l1, l2, l3, l4)
    # set up hyperparameters
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # train the model
    return - helpers.trainModel(
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


def evaluate_architecture(model, X, y, prep):
    
    results = model(X)
    results = results.detach().numpy()
    dataset = np.concatenate((X.detach().numpy(), y.detach().numpy()), axis = 1)
    dataset_pred = np.zeros(dataset.shape)
    dataset_pred[:, 3:6] = results
    dataset_pred[:, :3] = dataset[:, :3]
    dataset = prep.revert(dataset)
    dataset_pred = prep.revert(dataset_pred)
    prediction = dataset_pred[:, 3:6]
    true = dataset[:, 3:6]
    
    mae = mean_absolute_error(true, prediction)
    mse = mean_squared_error(true, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, prediction, multioutput = 'uniform_average')
    accuracy = np.average(1 - np.sqrt((prediction[:,0]-true[:,0])**2+(prediction[:,1]-true[:,1])**2+(prediction[:,2]-true[:,2])**2)/1732)
    
    print("\nTest set: Mean absolute error: {:.3f},  R2 score: {:.3f}".format(mae, r2))
    print("\t  Root mean squared error: {:.3f}\n".format(rmse))
    print("\t  Average Accuracy: {:.6f}\n".format(accuracy))
    print("\t  Note: the accuracy is measured as the euclidian distance") 
    print("\t        between true and prediction over a maximum euclidian error")
    print("\t        of 1732 (1000 in all 3 directions)")    
    
    return mae, mse, rmse, r2


def predict_hidden(dataset):
    # -------  Process data -------- #
    dataset = np.loadtxt(dataset)
    prep = Preprocessor(dataset)
    dataset = prep.apply(dataset)
    X, Y = torch.Tensor(dataset[:, 0:3]), torch.Tensor(dataset[:, 3:6])

    # ------- Instantiate model ----- #
    model = Net(3,3,237,248,106,115)

    # ----- Load our best model ------ #
    model.load_state_dict(torch.load('best_model_reg.pth'))

    # ----- Compute the output ------ #
    results = model(X)
    results = results.detach().numpy()

    # ----- Revert data processing ------ #
    dataset[:, 3:6] = results
    dataset = prep.revert(dataset)
    prediction = dataset[:, 3:6]

    return prediction # Returns a numpy array of shape (n_samples, 3)


if __name__ == "__main__":
    #main()
    pred = predict_hidden("FM_dataset.dat")
