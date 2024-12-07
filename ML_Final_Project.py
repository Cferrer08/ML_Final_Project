from tkinter import Y
from tracemalloc import take_snapshot
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



def GradientUpdate(alphas, dataTrain, stepSize, offset): #need to check if derivative update rule is correct

    alphaLen = len(alphas)
    trainRows, trainCols = np.shape(dataTrain)


    for i in range(alphaLen):
        classI = dataTrain[i, 0]
        featI = dataTrain[i, 1:]
        runTemp = 0
        for j in range(trainRows):
            runTemp += dataTrain[j, 0] * alphas[j] * BaseKernel(dataTrain[j, 1:], featI) #PLUS OFFSET, CHECK WHEN TO MULTIPLY ALPHA SUM AND X DATA POINT
            
        alphas[i] += 1 - classI * runTemp
        alphas[i] = max(0, alphas[i])
        
    return alphas
           
        


def BaseKernel(xI, xJ):
    return np.dot(xI, xJ)



def DualClassifier(alphas, dataTrain, offset, xToClass):
    
    classSum = 0
    for i in range(alphas): #if statement for alpha = 0
        classSum += alphas[i] * dataTrain[i, 0] * BaseKernel(dataTrain[i, 1:], xToClass)

    return classSum + offset


def DualLearner(dataTrain, iters, stepSize):
    
    trainRows, trainCols = np.shape(dataTrain)
    alphas = np.ones(trainRows)
    alphaLength = len(alphas)

    print(f'Begin Learner------------------\nNum Rows:{trainRows}\nNum Cols:{trainCols}')

    for step in range(iters):
        argMaxList = []
        alphaSum = 0
        print(f'\tIteration {step}------------------------------------')
        
        for i in range(alphaLength):
            yI = dataTrain[i, 0]
            xI = dataTrain[i, 1:]
            aI = alphas[i]
            tempSum = 0
            test_w = np.zeros(trainCols - 1)
            print(f'\t\tAlpha Value {i}: {aI}')

            for j in range(alphaLength): #add if statement for if alpha is 0
                tempSum += aI * alphas[j] * yI * dataTrain[j, 0] * BaseKernel(xI, dataTrain[j, 1:])
                test_w += alphas[j] * dataTrain[j, 0] * dataTrain[j, 1:]

            alphaSum += aI - (tempSum / 2)
            
        wTx = []
        print(f'\t\tClassify Test:')
        for x in range(alphaLength):
            wTx.append(np.dot(test_w, dataTrain[x, 1:]))
            print(f'\t\t\tData Point {x}: {wTx[x]}')

        offset = -(max(wTx[0:10]) + min(wTx[10:]))/2


        print(f'\tAlpha Sum: {alphaSum}\n\tAlphas:\n{alphas}\n\tTest W:\t{test_w}')
        argMaxList.append(alphaSum)
        alphas = GradientUpdate(alphas, dataTrain, stepSize, offset)
        

            


def main():
    
    test = pd.read_csv('SPECTF_test.csv')
    train = pd.read_csv('SPECTF_train.csv', header=None).to_numpy()
    

    classOne = train[0:40, 0:]
    classZero = train[40:, 0:]
    #print(classOne)
    #print(classZero)

    rngData = np.ndarray(shape = (10, 2), dtype= int, buffer=np.random.randint(low=0, high=4, size=20))
    rng2 = np.ndarray(shape = (10, 2), dtype= int, buffer=np.random.randint(low=6, high=10, size=20))
    rngData = np.append(rngData, rng2, axis=0)
    rngData = pd.DataFrame(rngData)
    rngData.insert(0, 'class', value=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1])
    print(rngData)
    
    plt.scatter(x=rngData[0], y=rngData[1])
    plt.show()


    iters = 10
    testing = DualLearner(rngData.to_numpy(), 10, 0.1) 
    

    #PROCESS-------
    #FIND ALL ALPHAS WITH EQUATION FROM PROJECT DESCRIPTION


if __name__ == '__main__':
    main()



