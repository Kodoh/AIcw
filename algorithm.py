import preProcessor
import random
import json
import math
import ast
import time
import numpy as np
import matplotlib.pyplot as plt
from math import e
trainingSet = preProcessor.training
validationSet = preProcessor.validation
testingSet = preProcessor.test
minMaxValuesTraining = preProcessor.minMaxValuesTraining
minMaxValuesTesting = preProcessor.minMaxValuesTesting

#base - hidden node object
class node: 
    def __init__(self,id):
        self.id = id
        self.s = 0
        self.u = 0
        self.fDiff = 0
        self.delta = 0
    def getU(self):
        return self.u
    def getId(self):
        return self.id
    def getfDiff(self):
        return self.fDiff
    def getDelta(self):
        return self.delta
    def calculateU(self):
        self.s = 0
        for i in range(1,N+1): 
            self.s += weights[(i,self.id)] * nodes[i-1].getU()
        self.s += weights[(0,self.id)]
        self.u = 1/(1+(e**(-1*self.s)))
        return self.u
    def calculateFdiff(self):
        self.fDiff = self.u * (1-self.u)
        return self.fDiff
    def calculateDelta(self):
        deltaOut = nodes[len(nodes)-1].getDelta()
        wiO = weights[(self.id,len(nodes))]
        self.delta = wiO*deltaOut*self.calculateFdiff()
        return self.delta

class outputNode(node):
    def __init__(self,id):
        super().__init__(id)
    def calculateDelta(self,realOut,penalty):
        self.delta = (realOut - self.u + penalty)*self.calculateFdiff()
        return self.delta
    def calculateU(self):
        self.s = 0
        for i in range(N+1,len(nodes)):  
            self.s += weights[(i,self.id)] * nodes[i-1].getU()
        self.s += weights[0,self.id]     
        self.u = 1/(1+(e**(-1*self.s)))   
        return self.u

class inputNode(node):
    def __init__(self,id):
        super().__init__(id)
    def setInput(self,newInput):
        self.u = newInput

weights = {}
nodes =  []
N = 5
lowest = float('inf')

def deStandardise(Sinput,minMax):
    deStandardisedOutput = ((Sinput - 0.1)/0.8)*(minMax[1]-minMax[0]) + minMax[0]

    return deStandardisedOutput


def activation (hiddenLayerNodes):
    nodeNum = hiddenLayerNodes + N + 1      #Total number of nodes in ANN
    outputIndex = hiddenLayerNodes+N        #Index in nodes[] of output node
    nodes.clear()                           #So when a new ANN is ran old nodes are not held in the array

    #Init for nodes[]
    for i in range(nodeNum):
        nodes.append(None)

    #Create objects to put in nodes[]
    for i in range(N):
        IN = inputNode(i+1)
        nodes[i] = IN
    
    #Create objects for hidden layer
    for i in range(N,outputIndex):
        hiddenNode = node(i+1)
        nodes[i] = hiddenNode

    #Create object for output node
    output = outputNode(nodeNum)
    nodes[outputIndex] = output

    weights.clear()                 #So when a new ANN is ran old weights are not held in the dictionary
    #Weights - input --> hidden
    for i in range(1,N+1):
        for j in range(N+1,len(nodes)):
            weights[(i,j)] = random.uniform(-2/N, 2/N)

    #biases
    for i in range(N+1,len(nodes)+1):
        weights[(0,i)] = random.uniform(-2/N, 2/N)

    #weights - hidden --> output
    for i in range(N+1,len(nodes)+1):
        if (i != len(nodes)):
            weights[(i,len(nodes))] = random.uniform(-2/N, 2/N)

def activationTesting(hiddenNodes):             #activation for testing set (dont need to set weights)
    nodeNum = hiddenNodes + N + 1             
    outputIndex = hiddenNodes +N
    nodes.clear()
    for i in range(nodeNum):
        nodes.append(None)
    
    for i in range(N):
        IN = inputNode(i+1)
        nodes[i] = IN

    for i in range(N,outputIndex):
        hiddenNode = node(i+1)
        nodes[i] = hiddenNode
    output = outputNode(nodeNum)
    nodes[outputIndex] = output
    

def mainLoop(epochs,momentum,BD,AN,WD,DataSet):
    p = 0.1                 #Learning rate
    omega = 0
    mse = 0
    for y in range(epochs):
        regulationParam = 1/(p*y+1)
        prev = mse
        mseSum = 0
        for x in range(len(DataSet)):          

            realOut = DataSet[x][6]             # Actual predictand

            if WD:                      
                omega = weightDepth()               #Weight depth

            forwardPass(DataSet,x)                  

            nodes[len(nodes)-1].calculateDelta(realOut,regulationParam*omega)           #For output node get delta

            backwardPass(momentum,p)                                                

            mseSum += (deStandardise(realOut,minMaxValuesTraining[len(minMaxValuesTraining)-1])-deStandardise(nodes[len(nodes)-1].getU(),minMaxValuesTraining[len(minMaxValuesTraining)-1]))**2 
            
        mse = mseSum/len(DataSet)
        if BD:
            p = boldDriver(mse,prev,p)          #Bold Driver

        if AN:
            p = annealing(y,epochs)            #Annealing

def forwardPass(DataSet,record):
    #INITIALISE

    for i in range(1,len(DataSet[record])-1):
        nodes[i-1].setInput(DataSet[record][i])
                    
    #Forward pass

    for i in range(5, len(nodes)):
        nodes[i].calculateU()

def boldDriver(mse,prev,p):
    if (mse - prev) > prev * 0.01:
        p *= 0.7
        if p < 0.01:
            p = 0.01

    elif (prev - mse) > prev * 0.01:
        p += p * 0.05
        if p > 0.5:
            p = 0.5
    return p

def weightDepth():
    sum1 = sum((weights[item]**2) for item in weights)
    omega = 1/(2*len(weights))*sum1
    return omega


def annealing(current,epochs):
    startLR = 0.1 
    endLR = 0.01
    return endLR + (startLR-endLR)*(1-(1/(1+e**(10-((20*current)/epochs)))))

def backwardPass(momentum,p):
    N = 5
    for i in range(N,len(nodes) -1):                #Get delta for hidden nodes
        nodes[i].calculateDelta()

    for i in weights:
        deltaValue = nodes[i[1]-1].getDelta()                  #Calculate new weights
        uValue = 1 if i[0] == 0 else nodes[i[0]-1].getU()
    
        if momentum:
            prev = weights[i]
            weights[i] = weights[i] + p * deltaValue * uValue
            diff = weights[i] - prev
            a = 0.9
            weights[i] = prev + p * deltaValue * uValue + a * diff
    
        else:
            weights[i] = weights[i] + p * deltaValue * uValue

def runValidation():            #Run ANN on validation set
    rmse = 0
    for i in range(len(validationSet)):
        realOut = validationSet[i][6]
        forwardPass(validationSet,i)
        rmse += (deStandardise(realOut,minMaxValuesTesting[len(minMaxValuesTesting)-1])-deStandardise(nodes[len(nodes)-1].getU(),minMaxValuesTesting[len(minMaxValuesTesting)-1]))**2 
    return (math.sqrt(rmse/len(validationSet)))


def runTesting():           #Run ANN on Testing set
    xData = []
    yData = []
    mseSum = 0
    meanOb = 0
    realMeanDiffSum  = 0
    msreSum = 0
    CE = 1
    for i in testingSet:
        meanOb += i[6]
    for i in range(len(testingSet)):
        minMaxPredictand = minMaxValuesTesting[len(minMaxValuesTraining)-1]
        realOut = deStandardise(testingSet[i][6],minMaxPredictand)
        realMeanDiffSum += (realOut - meanOb)**2
        xData.append(realOut)
        forwardPass(testingSet,i)
        modelled = deStandardise(nodes[len(nodes)-1].getU(),minMaxPredictand)
        yData.append(modelled)
        msreSum += (modelled-realOut/realOut)**2
        mseSum += (realOut-modelled)**2 

    plt.scatter(xData,yData)
    plt.axline((0, 0), slope=1)
    plt.ylabel('Modelled')              #Plot graph
    plt.xlabel('Actual')
    plt.title('Test ANN vs actual PAN')
    plt.show()
    mse = mseSum/len(testingSet)
    msre = msreSum/len(testingSet)
    CE -= (mseSum/realMeanDiffSum)
    rmse = math.sqrt(mse/len(testingSet))
    print(f"MSE - {mse} MSRE - {msre} CE - {CE} RMSE - {rmse}")


def setUp():
    lowest = float('inf')
    for i in range(100):
        start_time = time.time()
        epochs = random.randint(100,1000)
        hidden = random.randint(N//2,2*N)           #Create ANNs
        lowest = ANNcreate(hidden,epochs,lowest)
        print("--- %s seconds ---" % (time.time() - start_time))
    # Opening JSON file
    with open('bestANN.json', 'r') as openfile:         #Get best ANN to run on testing
    
        # Reading from json file
        json_object = json.load(openfile)
    

    weights.clear()
    weights.update(ast.literal_eval(json_object["weights"]))        #Set weights for testing set
    activationTesting(json_object["hidden"])
    runTesting()

def ANNcreate(hidden,epochs,low):
    momentum = random.randint(0, 1)
    boldDriver = random.randint(0, 1)
    annealing = random.randint(0, 1)            #Random improvments
    weight_decay = random.randint(0, 1)         


    activation(hidden) 
    mainLoop(epochs,momentum,boldDriver,annealing,weight_decay,trainingSet)

    RMSE = runValidation()              #Put ANN with best RMSE in bestANN.json
    if RMSE < low:
        ANN = {
            "hidden" : hidden,
            "epochs" : epochs,
            "momentum" : momentum,
            "boldDriver" : boldDriver,
            "annealing" : annealing,
            "WD" : weight_decay,
            "RMSE" : RMSE,
            "weights" :  str(weights)

        }
        json_object = json.dumps(ANN, indent=4)
        
        with open("bestANN.json", "w") as outfile:
            outfile.write(json_object)

        low = RMSE
        
    return low

setUp()
