import numpy as np
import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import optim

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)

# from illustrate import illustrate_results_ROI


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

    dataX, dataY = dataset[:, 0:3], dataset[:, 3:]
    # split the dataset
    dataX = torch.stack([torch.Tensor(i) for i in dataX])
    dataY = torch.stack([torch.Tensor(i) for i in dataY])
    # setup size ratios
    dataSize = int(dataset.shape[0])
    trainingSize = int(np.floor(dataSize * trainRatio))
    leftoverSize = int(dataSize - trainingSize)
    testSize = int(np.floor(leftoverSize * (testRatio/(1-trainRatio))))
    validationSize = leftoverSize - testSize

    trainX = dataX[:trainingSize]
    valX = dataX[trainingSize:trainingSize + validationSize]
    testX = dataX[trainingSize + validationSize:]
    trainY = dataY[:trainingSize]
    valY = dataY[trainingSize:trainingSize + validationSize]
    testY = dataY[trainingSize + validationSize:]

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

    model = Net(3, 4).to(device)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, numEpochs):

        train_loss, valid_loss = [], []
        # training part
        model.train()
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # print(output)
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
            train_loss), 4), "\tValid Loss: ", round(np.mean(valid_loss), 4))

    # test greatness of model
    model.eval()
    test_loss = 0
    num_right = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            for i in range(len(output)):
                left, right = torch.argmax(output[i]), torch.argmax(target[i])
                num_right += torch.eq(left, right).item()
            test_loss += loss_function(output, target).item()

    test_loss /= len(testloader.dataset)
    accuracy = num_right/ len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f} \tAccuracy: {:.4f}'.format(
        test_loss, accuracy))

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # illustrate_results_ROI(network, prep)


if __name__ == "__main__":
    main()
