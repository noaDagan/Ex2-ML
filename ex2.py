import numpy as np
import csv as csv
import random
import sys

# loss function of passive aggressive algorithm
def lossPA(x, y, weightPA, yHatPA):
    # calculate tha loss function and return max between zero
     temp = 1 - np.dot(weightPA[y], x) + np.dot(weightPA[yHatPA], x)
     if(temp > 0):
         return temp
     else:
         return 0

# The function recive 3 parameter and run over train x and y and train them
def prediction(trainingExample, label, testingExample):
    # open text of train x and y and create list of them
    trainX = open(trainingExample, "r")
    trainY = open(label, "r")
    reader = csv.reader(trainX, delimiter=",")
    trainY = list(trainY)
    dataMatrix = list(reader)
    # run over all the file and convert the first category for a number
    for index in dataMatrix:
        if index[0] == 'M':
            index[0] = 0.2
        elif index[0] == 'F':
            index[0] = 0.4
        elif index[0] == 'I':
            index[0] = 0.6
    # initialize weight array of all the algorithms
    weightP = np.array([[0 for j in range(8)] for i in range (3)])
    weightP = weightP.astype(np.float)
    weightS = np.array([[0 for j in range(8)] for i in range(3)])
    weightS = weightS.astype(np.float)
    weightPA = np.array([[0 for j in range(8)] for i in range(3)])
    weightPA = weightPA.astype(np.float)                              # run over all the line simultaneously in the file
    # initalize eta
    eta = 1
    # choose random x and in same line
    shffleW = list(zip(dataMatrix,trainY))
    random.shuffle(shffleW)
    dataMatrix,trainY = zip(*shffleW)
    # run over the algorithm in 100 iteration
    for i in range(100):
        # minimalize eta in all iteration
        eta = eta / 5
        # run over all the line in train x and train y
        for x,y in zip(dataMatrix,trainY):
            x = np.array(x)
            x = x.astype(np.float)
            # normalize the value in train x
            normX = x / np.linalg.norm(x)
            # calculate YHat for all the algorithms
            yHatP = np.argmax(np.dot(weightP, normX))
            yHatS = np.argmax(np.dot(weightS, normX))
            yHatPA = np.argmax(np.dot(weightPA, normX))
            y = int(float(y))
            # check if the prediction match the expected and update for Perceptron algorithm the values
            if y != yHatP:
                weightP[y, :] = weightP[y, :] + eta * normX
                weightP[yHatP, :] = weightP[yHatP, :] - eta * normX
            # check if the prediction match the expected and update for Svm algorithm the values
            if y != yHatS:
                weightS[y, :] = (1 - 0.075 * eta) * weightS[y, :] + eta * normX
                weightS[yHatS, :] = (1 - 0.075 * eta) * weightS[yHatS, :] - eta * normX
                # update the third number
                if (((y == 0) & (yHatS == 1)) | ((y == 1) & (yHatS == 0))):
                    weightS[2, :] = (1 - 0.075 * eta) * weightS[2, :]
                elif (((y == 0) & (yHatS == 2)) | ((y == 2) & (yHatS == 0))):
                    weightS[1, :] = (1 - 0.075 * eta) * weightS[1, :]
                elif (((y == 1) & (yHatS == 2)) | ((y == 2) & (yHatS == 1))):
                    weightS[0, :] = (1 - 0.075 * eta) * weightS[0, :]
            # check if the prediction match the expected and update for Passive Agressive algorithm the values
            if y != yHatPA:
                # calculate theo
                res = lossPA(normX,y,weightPA,yHatPA) / (pow(np.linalg.norm(normX),2) * 2)
                weightPA[y, :] = weightPA[y, : ] + res * normX
                weightPA[yHatPA, :] = weightPA[yHatPA, :] - res * normX

    # open the test file and convert to array
    test = open(testingExample, "r")
    reader = csv.reader(test, delimiter=",")
    test = list(reader)
    # run over all the file and convert the first category for a number
    for index in test:
         if index[0] == 'M':
             index[0] = 0.2
         elif index[0] == 'F':
             index[0] = 0.4
         elif index[0] == 'I':
             index[0] = 0.6
    test = np.array(test)
    test = test.astype(np.float)
    # run over all the test file and print the prediction after the train of the algorithms
    for t in test:
        # normalize the value in the test
        norm = t / np.linalg.norm(t)
        yHatP = np.argmax(np.dot(weightP, norm))
        yHatS = np.argmax(np.dot(weightS, norm))
        yHatPA = np.argmax(np.dot(weightPA, norm))
        print("perceptron: ",yHatP, ", svm: ", yHatS, ", pa: ",yHatPA,sep="")

def main():
    commandArg = sys.argv
    prediction(commandArg[1],commandArg[2],commandArg[3])
main()