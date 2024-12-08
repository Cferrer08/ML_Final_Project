from tracemalloc import stop
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



def GradientUpdate(alphas, alphaLength, dataTrain, stepSize): #need to check if derivative update rule is correct
    
    deltaAlpha = []
    
    for i in range(alphaLength):
        featI = dataTrain[i, 1:]
        tempSum = 0
        
        for j in range(alphaLength):
            tempSum += alphas[j] * dataTrain[j, 0] * LinearKernel(dataTrain[j, 1:], featI)
                
        #tempSum = BaseKernel(tempSum, featI)
            
        tempSum = stepSize * (1 - (dataTrain[i, 0] * tempSum))
        deltaAlpha.append(tempSum)                

    #print(f'Change in Alphas:\n{deltaAlpha}')

    for a in range(alphaLength):
        alphas[a] += deltaAlpha[a]
        alphas[a] = max(0, alphas[a])

    return alphas
           
       #calculate kernelized matrix of dot products
    # kernelMatrix = np.ndarray(shape=(alphaLength, alphaLength))
    # ones = np.ones(alphaLength)
    # for i in range(alphaLength):
    #     classI = dataTrain[i, 0]
    #     featI = dataTrain[i, 1:]
    #     for j in range(alphaLength):
    #         kernelMatrix[i, j] = classI * dataTrain[j, 0] * BaseKernel(featI, dataTrain[j, 1:])

    # print(f'Kernel Dot Prodect Matrix:\n{kernelMatrix}\n\n')
    
    # alphas += stepSize * (ones - np.matmul(kernelMatrix, alphas))
    
    # for a in range(alphaLength):
    #     alphas[a] = max(0, alphas[a]) 


def LinearKernel(xI, xJ):
    return np.dot(xI, xJ)

def QuadraticKernel(xI, xJ):
    square = np.dot(xI, xJ)
    return (square+1) * (square+1)


def DualClassifier(alphas, dataTrain, offset, xToClass):
    
    classSum = 0
    for i in range(alphas): #if statement for alpha = 0
        classSum += alphas[i] * dataTrain[i, 0] * LinearKernel(dataTrain[i, 1:], xToClass)

    return classSum + offset


def DualLearner(dataTrain, iters, stepSize):
    
    trainRows, trainCols = np.shape(dataTrain)
    alphas = np.ones(trainRows)
    alphaLength = len(alphas)
    objFunc = []
    
    for step in range(iters):
        objFuncSum = 0
        print(f'Iteration {step}--------------------------------------------------------------------------------')

        for i in range(alphaLength):
            classI = dataTrain[i, 0]
            featI = dataTrain[i, 1:]
            alphaI = alphas[i]
            tempSum = 0

            for j in range(alphaLength):
                tempSum += classI * dataTrain[j, 0] * alphaI * alphas[j] * LinearKernel(dataTrain[j, 1:], featI)
            
            #tempSum = BaseKernel(tempSum, featI)
            #print(f'Kernel Dot Prod: {tempSum}')
            tempSum = max(0, tempSum)
            objFuncSum += alphaI - (tempSum / 2)
            objFunc.append(objFuncSum)
        #print(f'Object Function Sum: {objFunc[step]}')
        
        alphas = GradientUpdate(alphas, alphaLength, dataTrain, stepSize)
        #print(f'Post Gradient Alphas:\n{alphas}\n\n\n')

    return alphas

    

def GetW(alphas, dataTrain):
    
    w = np.zeros(len(dataTrain[0]) -1)
    for i in range(len(alphas)):
        w += alphas[i] * dataTrain[i, 0] * dataTrain[i, 1:]

    return w

def main():
    
    test = pd.read_csv('SPECTF_test.csv')
    train = pd.read_csv('SPECTF_train.csv', header=None).to_numpy()
    

    classOne = train[0:40, 0:]
    classZero = train[40:, 0:]
    #print(classOne)
    #print(classZero)

    rngData = np.random.normal(0.5, 0.2, (10, 2))
    rngData2 = np.random.normal(-0.5, 0.2, (10, 2))
    rngData = np.append(rngData, rngData2, axis=0)

    rngData = pd.DataFrame(rngData)
    rngData.insert(0, 'class', value=1)
    rngData = rngData.to_numpy()
    for i in range(10):
        rngData[i+10, 0] *= -1

    print(rngData)
    #rngData = pd.DataFrame([[1, 8, 8], [1, 6, 6], [-1, 1, 1], [-1, 4, 4]])
    #print(rngData)

    plt.scatter(x=rngData[:, 1], y=rngData[:, 2])
    plt.show()


    testingAlphas = DualLearner(rngData, 50, 0.1) 
    print(f'Alphas:{testingAlphas}')
    w = GetW(testingAlphas, rngData)
    print(f'W:\n{w}')


    #PROCESS-------
    #FIND ALL ALPHAS WITH EQUATION FROM PROJECT DESCRIPTION


if __name__ == '__main__':
    main()



