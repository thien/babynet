import numpy as np
import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim


def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    trainRatio = 0.8
    testRatio  = 0.1
    numEpochs  = 50
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

    # check for gpu.
    device = "cpu"
    print(device)
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
            x = self.f(x)
            return x

    model = Net(3,3).to(device)

    loss_function = nn.MSELoss()
    optimizer     = optim.Adam(model.parameters())

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

        print("Epoch:", epoch, "\tTraining Loss: ", round(np.mean(
            train_loss),4), "\tValid Loss: ", round(np.mean(valid_loss),4))

    # test greatness of model
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += loss_function(output, target).item()

    test_loss /= len(testloader.dataset)

    print('\nTest set: Average loss: {:.4f}'.format(
        test_loss))

        
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # illustrate_results_FM(network, prep)


if __name__ == "__main__":
    main()