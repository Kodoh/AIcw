import preProcessor
import random
from math import e
trainingSet = preProcessor.training
validationSet = preProcessor.validation
testingSet = preProcessor.test
minMaxValuesTraining = preProcessor.minMaxValuesTraining
minMaxValuesTesting = preProcessor.minMaxValuesTesting


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
    def calculateS(self):
        self.s = 0
        for i in range(1,n+1): 
            self.s += weights[(i,self.id)] * nodes[i-1].getU()
        self.s += weights[(0,self.id)]
        return self.s
    def getDelta(self):
        return self.delta
    def calculateU(self):
        self.u = 1/(1+(e**(-1*self.calculateS())))
        return self.u
    def calculateFdiff(self):
        self.fDiff = self.u * (1-self.u)
        return self.fDiff

    def calculateDelta(self,weights):
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
    
    def calculateS(self):
        self.s = 0
        for i in range(n+1,len(nodes)):
            self.s += weights[(i,self.id)] * nodes[i-1].getU()
        self.s += weights[0,self.id]        
        return self.s

class inputNode(node):
    def __init__(self,id):
        super().__init__(id)
    def setInput(self,newInput):
        self.u = newInput


class ANN():
    def __init__(self,hidden,epochs,momentum,boldDriver,annealing,WD,MSE,weights):
        self.hidden = hidden
        self.epochs = epochs
        self.momentum = momentum
        self.boldDriver = boldDriver
        self.annealing = annealing
        self.WD = WD
        self.MSE = MSE
        self.weights = weights

    def getHidden(self):
        return self.hidden
    def getEpochs(self):
        return self.epochs
    def getMomentum(self):
        return self.momentum
    def getBoldDriver(self):
        return self.boldDriver
    def getAnnealing(self):
        return self.annealing
    def getWD(self):
        return self.WD
    def getMSE(self):
        return self.MSE
    def getWeights(self):
        return self.weights


weights = {}
nodes =  []
n = 5

def deStandardise(Sinput,minMax):
    deStandardisedOutput = ((Sinput - 0.1)/0.8)*(minMax[1]-minMax[0]) + minMax[0]

    return deStandardisedOutput


def activation (hiddenLayerNodes):
    nodeNum = hiddenLayerNodes + n + 1
    outputIndex = hiddenLayerNodes+n
    for i in range(nodeNum-len(nodes)):
        nodes.append(None)


    for i in range(n):
        IN = inputNode(i+1)
        nodes[i] = IN
    
    for i in range(n,outputIndex):
        hiddenNode = node(i+1)
        nodes[i] = hiddenNode

    output = outputNode(nodeNum)
    nodes[outputIndex] = output

    for i in range(1,n+1):
        for j in range(n+1,len(nodes)):
            weights[(i,j)] = random.uniform(-2/n, 2/n)

    #biases
    for i in range(n+1,len(nodes)+1):
        weights[(0,i)] = random.uniform(-2/n, 2/n)

    for i in range(n+1,len(nodes)+1):
        if (i != len(nodes)):
            weights[(i,len(nodes))] = random.uniform(-2/n, 2/n)
    print(weights)

def activationTesting(hiddenLayerNodes):
    nodeNum = hiddenLayerNodes + n + 1
    outputIndex = hiddenLayerNodes+n
    for i in range(nodeNum-len(nodes)):
        nodes.append(None)

    for i in range(n,outputIndex):
        hiddenNode = node(i+1)
        nodes[i] = hiddenNode
    output = outputNode(nodeNum)
    nodes[outputIndex] = output
    

xData = []
def mainLoop(epochs,momentum,BD,AN,WD,DataSet):
    p = 0.1 
    omega = 0
    mse = 0
    for y in range(epochs):
        regulationParam = 1/(p*y+1)
        global yData
        yData = []
        mseSum = 0
        prev = mse
        for x in range(len(DataSet)):          

            realOut = DataSet[x][6]

            if WD:
                omega = weightDepth()

            forwardPass(DataSet,x)

            nodes[len(nodes)-1].calculateDelta(realOut,regulationParam*omega)

            backPropogate(momentum,p)

            mseSum += (deStandardise(realOut,minMaxValuesTraining[len(minMaxValuesTraining)-1])-deStandardise(nodes[len(nodes)-1].getU(),minMaxValuesTraining[len(minMaxValuesTraining)-1]))**2 

        mse = mseSum/len(DataSet)
        if BD:
            p = boldDriver(mse,prev,p)

        if AN:
            p = annealing(y,epochs)

        print(mse)
        

def forwardPass(DataSet,x):
    #INITIALISE

    for i in range(1,len(DataSet[x])-1):
        nodes[i-1].setInput(DataSet[x][i])
                    
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


def annealing(y,epochs):
    startLR = 0.1 
    endLR = 0.01
    return endLR + (startLR-endLR)*(1-(1/(1+e**(10-((20*y)/epochs)))))

def backPropogate(momentum,p):
    n = 5
    for i in range(n,len(nodes) -1):
        nodes[i].calculateDelta(weights)

    for i in weights:
        deltaValue = nodes[i[1]-1].getDelta()
        uValue = 1 if i[0] == 0 else nodes[i[0]-1].getU()
    
        if momentum:
            a = 0.9
            prev = weights[i]
            weights[i] = weights[i] + p * deltaValue * uValue
            diff = weights[i] - prev
            weights[i] = prev + p * deltaValue * uValue + a * diff
    
        else:
            weights[i] = weights[i] + p * deltaValue * uValue


def runValidation():
    mse = 0
    for i in range(len(validationSet)):
        realOut = validationSet[i][6]
        forwardPass(validationSet,i)
        print(str(realOut) + ' ' + str(nodes[len(nodes)-1].getU()))
        mse += (deStandardise(realOut,minMaxValuesTesting[len(minMaxValuesTesting)-1])-deStandardise(nodes[len(nodes)-1].getU(),minMaxValuesTesting[len(minMaxValuesTesting)-1]))**2 
    return (mse/len(validationSet))


def runTesting():
    mse = 0
    for i in range(len(testingSet)):
        realOut = testingSet[i][6]
        forwardPass(testingSet,i)
        print(str(realOut) + ' ' + str(nodes[len(nodes)-1].getU()))
        mse += (deStandardise(realOut)-deStandardise(nodes[len(nodes)-1].getU()))**2 
    return (mse/len(testingSet))


def variation():
    ANNs = []
    # for i in range(5,20,5):
    #     for j in range(100,1000,100):
    #         ANNcreate(i,j,ANNs)

    # lowest = float('inf')
    # testingANN = None
    # for i in ANNs:
    #     if i.getMSE() < lowest:
    #         lowest = i.getMSE()
    #         testingANN = i

    # weights = testingANN.getWeights()
    # print(weights)
    # activationTesting(testingANN.getHidden(),5)
    # runTesting()
    # print(testingANN.getHidden())
    # print(testingANN.getEpochs())
    # print(testingANN.getMomentum())
    # print(testingANN.getBoldDriver())
    # print(testingANN.getAnnealing())
    # print(testingANN.getWD())
    # print(testingANN.getMSE())
    # print(testingANN.getWeights())

    activation(5) 
    mainLoop(1,0,0,0,0,trainingSet)
    for i in range(5000):
        for j in range(len(trainingSet)):
            forwardPass(trainingSet,j)
    print(weights)



def ANNcreate(i,j,ANNs):

    #NO IMPROVMENT
    activation(i,5) 
    mainLoop(j,0,0,0,0,trainingSet)
    ANNSample = ANN(i,j,0,0,0,0,runValidation(),weights)
    ANNs.append(ANNSample)


    #WD
    activation(i,5) 
    mainLoop(j,0,0,0,1,trainingSet)
    ANNSample = ANN(i,j,0,0,0,1,runValidation(),weights)
    ANNs.append(ANNSample)
    
    #AN
    activation(i,5) 
    mainLoop(j,0,0,1,0,trainingSet)
    ANNSample = ANN(i,j,0,0,1,0,runValidation(),weights)
    ANNs.append(ANNSample)

    #BD
    activation(i,5) 
    mainLoop(j,0,1,0,0,trainingSet)
    ANNSample = ANN(i,j,0,1,0,0,runValidation(),weights)
    ANNs.append(ANNSample)

    #MM
    activation(i,5) 
    mainLoop(j,1,0,0,0,trainingSet)
    ANNSample = ANN(i,j,1,0,0,0,runValidation(),weights)
    ANNs.append(ANNSample)



    
    #change this value
    # add momentum


variation()
# xData = []
# for i in trainingSet:
#     xData.append(i[6])

# plt.scatter(xData, yData)
# plt.xlabel('epochs')
# plt.ylabel('MSE')
# plt.title('Error ')
# plt.show()