import numpy as np
import pandas as pd



def BaseKernel(xI, xJ):
    return np.dot(xI, xJ)




def DualClassifier(alphas, dataTrain, offset, xToClass):
    
    classSum = 0
    for i in range(alphas): #if statement for alpha = 0
        classSum += alphas[i] * dataTrain[i, 0] * BaseKernel(dataTrain[i, 1:], xToClass)

    return classSum + offset


def DualLearner(dataTrain, iters):
    
    trainRows, trainCols = np.size(dataTrain)
    alphas = np.ones(trainRows)
    alphaLength = len(alphas)

    for step in range(iters):
        sumAlphas = np.sum(alphas)
        argMaxList = []
        
        for i in range(alphaLength):
            yI = dataTrain[i, 0]
            xI = dataTrain[i, 1:]
            aI = alphas[i]
            sumDual = 0
            for j in range(alphaLength): #add if statement for if alpha is 0
                sumDual += yI * dataTrain[j, 0] * aI * dataTrain[j, 0] * BaseKernel(xI, dataTrain[j, 1:])

            argMaxList.append(sumAlphas - (sumDual / 2))

    #update rule?


def main():
    
    test = pd.read_csv('SPECTF_test.csv')
    train = pd.read_csv('SPECTF_train.csv')
    
    iters = 100
    DualLearner(train, iters) 
    #PROCESS-------
    #FIND ALL ALPHAS WITH EQUATION FROM PROJECT DESCRIPTION


if __name__ == '__main__':
    main()



