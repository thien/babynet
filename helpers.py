import numpy as np
import torch
import torch.utils.data as utils
import torch.nn.functional as F


"""
This file contains helper functions
used for part 2 and part 3 of the coursework.
"""

def loadTrainingData(dataX,dataY, trainRatio, testRatio, batchsize):
    """
    Deals with processing of the dataset.
    """

    # split the dataset
    dataX = torch.stack([torch.Tensor(i) for i in dataX])
    dataY = torch.stack([torch.Tensor(i) for i in dataY])

    # setup size ratios
    dataSize       = int(dataX.shape[0])
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

    return (trainloader, validloader, testloader)


def trainModel(model, optimizer, loss_function, numEpochs, trainloader, validloader, verbose=True):
    """
    Trains the model.
    """
    for epoch in range(1, numEpochs):

        train_loss, valid_loss = [], []
        # training part
        model.train()
        for data, target in trainloader:
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
            output = model(data)
            loss = loss_function(output, target)
            valid_loss.append(loss.item())

        if verbose:
            print("Epoch:", epoch, "\tTraining Loss: ", round(np.mean(
            train_loss), 4), "\tValid Loss: ", round(np.mean(valid_loss), 4))
    
    return np.mean(valid_loss)


def testModel(model, loss_function, numEpochs, testloader, checkAccuracy=False, verbose=True):
    """
    Used to test the performance of a model.
    """
    model.eval()
    test_loss = 0
    num_right = 0
    with torch.no_grad():
        for data, target in testloader:
            # evaluate and measure loss
            output = model(data)
            test_loss += loss_function(output, target).item()

            if checkAccuracy:
                # sum up batch loss
                for i in range(len(output)):
                    left, right = torch.argmax(output[i]), torch.argmax(target[i])
                    num_right += torch.eq(left, right).item()

    # get mean loss
    test_loss /= len(testloader.dataset)

    if verbose:
        if checkAccuracy:
            accuracy = num_right/ len(testloader.dataset)
            print('\nTest set: Average loss (on normalised data): {:.4f} \tAccuracy: {:.4f}'.format(
                test_loss, accuracy))
        else:
            print('\nTest set: Average loss (on normalised data): {:.4f}'.format(
            test_loss))

    return test_loss


def saveModel(model, filepath):
    torch.save(model, filepath)

def loadModel(filepath):
    return torch.load(filepath)