import numpy as np
import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim
import helpers


from illustrate import illustrate_results_ROI

def main():
    dataset = np.loadtxt("ROI_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    np.random.shuffle(dataset)

    trainRatio = 0.8
    testRatio = 0.1
    numEpochs = 20
    batchsize = 64

    X, Y = dataset[:, 0:3], dataset[:, 3:]

    trainloader, validloader, testloader = helpers.loadTrainingData(
        X, Y, trainRatio, testRatio, batchsize)

    # set up the model
    class Net(nn.Module):
        def __init__(self, in_size, out_size):
            super(Net, self).__init__()
            self.a = nn.Linear(in_size, 20)
            self.d = nn.Linear(20, 15)
            self.b = nn.Linear(15, 10)
            self.e = nn.Linear(10, 7)
            self.f = nn.Linear(7, out_size)

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)

        def forward(self, x):
            x = F.relu(self.a(x))
            x = F.relu(self.d(x))
            x = F.relu(self.b(x))
            x = F.relu(self.e(x))
            x = self.f(x)
            return x

    # initiate the model
    model = Net(3, 4)

    # set up hyperparameters
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    # train the model
    helpers.trainModel(model, optimizer, loss_function, numEpochs, trainloader, validloader)
    # test the model
    helpers.testModel(model, loss_function, numEpochs, testloader, checkAccuracy=True)

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    illustrate_results_ROI(model)


if __name__ == "__main__":
    main()
