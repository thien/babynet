import numpy as np
import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim
import helpers
from illustrate import illustrate_results_ROI
from bayes_opt import BayesianOptimization
# from keras.utils.np_utils import to_categorical  
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import copy 
#######################################################################
#                       ** START OF YOUR CODE **
#######################################################################
# otherwise what's the point of training the dataset?
seed = 1337
np.random.seed(seed)
torch.manual_seed(seed)

# default parameters for the model.
defaults = {
    'l1':20, 
    'l2':15, 
    'l3':10, 
    'l4':7,
    'lr':1e-3,
    'save':False, # determines whether to save the model to file or not.
    'epochs':20,
    'filepath': './Model/roi_model.pt'
}

# set up the model
class Net(nn.Module):
    def __init__(self, in_dim, out_dim, l1=20, l2=15, l3=10, l4=7):
        super(Net, self).__init__()
        self.l1  = nn.Linear(in_dim, l1)
        self.l2  = nn.Linear(l1, l2)
        self.l3  = nn.Linear(l2, l3)
        self.l4  = nn.Linear(l3, l4)
        self.out = nn.Linear(l4, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.out(x)
        return x


def predict_hidden(hidden_dataset):
    # load dataset
    dataset = np.loadtxt(hidden_dataset)
    dataset = torch.Tensor(dataset)
    # split dataset
    X, y = dataset[:, 0:3], dataset[:, 3:]
    # load model
    print("Loading model from:",defaults['filepath'])
    model = torch.load(defaults['filepath'])
    # set to evaluate only
    model.eval()
    # predict Y
    pred = model(X).detach().numpy()
    # onehot the output
    pred = pred > 0.5
    pred = pred.astype(float)
    # print results
    print_stats(y.numpy(), pred)
    # return what is specified in the question
    return pred


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
    dataset = np.loadtxt("ROI_dataset.dat")
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
    """
    Performs a hyperparameter search on the model
    based on the dataset provided in the assignment.

    """
    print("Initiating Bayesian Optimisation...")
    # initiate parameters
    params = {
        'l1': (3, 30), 
        'l2': (3, 20), 
        'l3': (3, 15),
        'l4': (3, 10),
        'lr': (0.0001, 0.001)
    }

    # initiate optimiser
    bayes = BayesianOptimization(
                f=blackbox,
                pbounds=params,
                random_state=seed)

    # search hyperparameter space
    bayes.maximize(init_points=4,n_iter=30)
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
    print("Saved ROI model as:", arguments['filepath'])


def evaluate_architecture(model, X, y):
    """
    This is separate to learn_FM as the metrics used to 
    evaluate the model is entirely different.
    """
    results = model(X)
    results = results.detach().numpy()
    dataset = np.concatenate((X.detach().numpy(), y.detach().numpy()), axis = 1)
    dataset_pred = np.zeros(dataset.shape)
    dataset_pred[:, 3:] = results
    dataset_pred[:, :3] = dataset[:, :3]
    prediction = dataset_pred[:, 3:]
    # onehot the output
    prediction = prediction > 0.5
    prediction = prediction.astype(float)
    true = dataset[:, 3:]
    print_stats(true, prediction)


def print_stats(true, prediction):
    """
    Prints statistics of the prediction against
    the true value.
    """
    accuracy = accuracy_score(true, prediction)
    precision = precision_score(true, prediction, average='micro')
    recall = recall_score(true, prediction, average='micro')
    f1 = f1_score(true, prediction, average='micro')
    print("Test Set: Accuracy: {:.3f}  Precision: {:.3f}  Recall: {:.3f}  F1: {:.3f}".format(accuracy,precision,recall,f1))

    # return mae, mse, rmse, r2
#######################################################################
#                       ** END OF YOUR CODE **
#######################################################################

def run(verbose=True, showBot=False, args=defaults):
    dataset = np.loadtxt("ROI_dataset.dat")
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
    helpers.testModel(model, loss_function, numEpochs, testloader, checkAccuracy=False, verbose=verbose)

    # evaluate model
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

    evaluate_architecture(model, torch.Tensor(x_store), torch.Tensor(y_store))
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    if showBot:
        illustrate_results_ROI(model)

    return val_loss


if __name__ == "__main__":
    # run(verbose=True, showBot=True)
    # optimise()
    predict_hidden("ROI_dataset.dat")