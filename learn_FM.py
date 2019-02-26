import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim
import helpers
from nn_lib import Preprocessor
from illustrate import illustrate_results_FM

# create the model.
class Net(nn.Module):
    def __init__(self, in_size, out_size):
        super(Net, self).__init__()
        self.a = nn.Linear(in_size,20)
        self.d = nn.Linear(20,10)
        self.e = nn.Linear(10,7)
        self.f = nn.Linear(7, out_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = F.relu(self.a(x))
        x = F.relu(self.d(x))
        x = F.relu(self.e(x))
        # we do not have an activation
        # function on the last layer deliberately.
        x = self.f(x)
        return x
    
    
def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    np.random.shuffle(dataset)
    
    trainRatio = 0.8
    testRatio  = 0.1
    numEpochs  = 20
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
    model = Net(3,3)

    # load hyperparameters
    loss_function = nn.MSELoss()
    optimizer     = optim.Adam(model.parameters())

    # train the model
    helpers.trainModel(model, optimizer, loss_function, numEpochs, trainloader, validloader)
    # evaluate the model
    helpers.testModel(model, loss_function, numEpochs, testloader)
    
    torch.save(model.state_dict(), 'best_model_reg.pth')
    
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # data is normalised in this function
    illustrate_results_FM(model, prep)


def predict_hidden(dataset):
    # -------  Process data -------- #
    prep = Preprocessor(dataset)
    dataset = prep.apply(dataset)
    X, Y = torch.Tensor(dataset[:, 0:3]), torch.Tensor(dataset[:, 3:6])

    # ------- Instantiate model ----- #
    model = Net(3,3)

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
    main()
