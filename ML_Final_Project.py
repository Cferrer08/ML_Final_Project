from tracemalloc import stop
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import exp, inf


def GradientUpdate(alphas, alphaLength, dataTrain, stepSize, C): #need to check if derivative update rule is correct
    
    deltaAlpha = []
    
    for i in range(alphaLength):
        featI = dataTrain[i, 1:]
        tempSum = 0
        
        for j in range(alphaLength):
            tempSum += alphas[j] * dataTrain[j, 0] * RBFKernel(dataTrain[j, 1:], featI, 10) #RBFKernel(dataTrain[j, 1:], featI, 0.1)  #dataTrain[j, 1:] #
                
        #tempSum = LinearKernel(tempSum, featI)
            
        tempSum = stepSize * (1 - (dataTrain[i, 0] * tempSum))
        deltaAlpha.append(tempSum)                

    print(f'Change in Alphas:\n{deltaAlpha}')

    for a in range(alphaLength):
        alphas[a] += deltaAlpha[a]
        alphas[a] = max(0, alphas[a])
        alphas[a] = min(alphas[a], C)

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


def RBFKernel(xI, xJ, variance):
    
    diff = xI - xJ
    #diff_2 = np.dot(diff, diff)
    #double_var = 2 * variance
    #power = float(diff_2)/double_var
    #print(f'Difference: {diff}\nDiff Squared:{type(diff_2)} {float(diff_2)}, 2*Variance: {type(double_var)} {double_var}\nPower: {power}')
    #e_xp = exp(power) 
    return exp( np.dot(diff, diff) / (2 * variance) )
    

def DualClassifier(alphas, dataTrain, offset, xToClass):
    
    classSum = 0
    for i in range(alphas): #if statement for alpha = 0
        classSum += alphas[i] * dataTrain[i, 0] * LinearKernel(dataTrain[i, 1:], xToClass)

    return classSum + offset



def DualLearner(dataTrain, iters, stepSize, C=inf):
    
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
                tempSum += classI * dataTrain[j, 0] * alphaI * alphas[j] * RBFKernel(dataTrain[j, 1:], featI, 10) #RBFKernel(dataTrain[j, 1:], featI, 0.1) # #dataTrain[j, 1:]
            
            #tempSum = LinearKernel(tempSum, featI)
            #print(f'Kernel Dot Prod: {tempSum}')
            #tempSum = max(0, tempSum)
            objFuncSum += alphaI - (tempSum / 2)
            objFunc.append(objFuncSum)
        print(f'Object Function Sum: {objFunc[step]}')
        
        alphas = GradientUpdate(alphas, alphaLength, dataTrain, stepSize, C)
        print(f'Post Gradient Alphas:\n{alphas}\n\n\n')

    return alphas

    

def GetW(alphas, dataTrain):
    
    w = np.zeros(len(dataTrain[0]) -1)
    for i in range(len(alphas)):
        w += alphas[i] * dataTrain[i, 0] * dataTrain[i, 1:]

    return w

def GetB(w, data):
    
    wTx = []
    for i in range(len(data)):
        wTx.append(np.dot(w, data[i, 1:]))
        
    classOne = min(wTx[0:11])
    classNeg = max(wTx[10:])
    
    return -(classOne + classNeg) / 2
        


def main():
    
    test = pd.read_csv('SPECTF_test.csv', header=None).to_numpy()
    train = pd.read_csv('SPECTF_train.csv', header=None).to_numpy()
    #classOne = train[0:40, 0:]
    #classZero = train[40:, 0:]
    #print(classOne)
    #print(classZero) 

    test = test[187:, :]
    train = train[80:, :]

    print(test)
    print(train)
    stop    

    trainAlphas = DualLearner(train, 50, 0.01)
    w = GetW(trainAlphas, train)
    b = GetB(w, train)

    print(f'Alphas:\n\n{trainAlphas}\n\nW:{w}\n\nb:{b}\n\n')

    testRes = []
    for i in range(len(test)):
        testRes.append(np.sign(np.dot(w, test[i, 1:]) + b))
    print(testRes)
    
    testClasses = np.array(test[:, 0])
    posCount = 0
    negCount = 0
    for i in range(len(testClasses)):
        if testClasses[i] == 1:
            posCount += 1
        else:
            negCount += 1
            testClasses[i] = -1

    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0
    
    for i in range(len(testClasses)):
        if testClasses[i] == 1 and testRes[i] == 1:
            truePos += 1
        elif testClasses[i] == -1 and testRes[i] == -1:
            trueNeg += 1
        elif testClasses[i] == -1 and testRes[i] == 1:
            falsePos += 1
        else:
            falseNeg += 1
        
    print(f'Total Instances: {len(testClasses)}\nPositive Class:{posCount}\nNegative Class:{negCount}')
    print(f'T Pos: {truePos}\nT Neg: {trueNeg}\nF Pos: {falsePos}\nF Neg: {falseNeg}') 

    # #TRAINING TEST DATA
    # rngData = np.random.normal(0.5, 0.2, (10, 2))
    # rngData2 = np.random.normal(-0.5, 0.2, (10, 2))
    # rngData = np.append(rngData, rngData2, axis=0)

    # rngData = pd.DataFrame(rngData)
    # rngData.insert(0, 'class', value=1)
    # rngData = rngData.to_numpy()
    # for i in range(10):
    #     rngData[i+10, 0] *= -1

    # print(rngData)
    # plt.scatter(x=rngData[:, 1], y=rngData[:, 2])
    # plt.show()
    
    # #get params
    # testingAlphas = DualLearner(rngData, 50, 0.1) 
    # print(f'Alphas:{testingAlphas}')
    # w = GetW(testingAlphas, rngData)
    # b = GetB(w, rngData)
    # print(f'W:\n{w}')
    # print(f'b: {b}')
    
    # #testing test data
    # testData = np.random.normal(0.5, 0.2, (5,2))
    # t2 = np.random.normal(-0.5, 0.2, (5,2))
    # testData = np.append(testData, t2, axis=0)
    # print(testData)
    # testData = pd.DataFrame(testData)
    # testData.insert(0, 'class', value=1)
    # testData = testData.to_numpy()
    # for i in range(5):
    #     testData[i+5, 0] *= -1
    # print(testData)

    


    #PROCESS-------
    #FIND ALL ALPHAS WITH EQUATION FROM PROJECT DESCRIPTION


if __name__ == '__main__':
    main()



