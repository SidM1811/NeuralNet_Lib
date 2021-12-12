import math
class Perceptron:
    def __init__(self,bias,numInput):
        self.bias=bias
        self.weight=[0.0]*numInput
    def activation(x):
        return 1/(1+math.exp(-x))
    def output(self,arr):
        output=0.0
        for i in range(0,min(len(arr),len(self.weight))):
            output+=self.weight[i]*arr[i]
        return Perceptron.activation(output)

class Layer:
    def __init__(self,prevLayer,currLayer,delta):
        self.array=[Perceptron(0,prevLayer)]*currLayer
        self.delta=[0.0]*currLayer
        self.ouput=[0.0]*currLayer
    def feedForward(self,inArr):
        for i in range(0,self.array):
            self.output[i]=self.array[i].output(inArr)
        return self.output
    def calcDelta(self,nextLayer,o):
        for i in range(0,len(self.delta)):
            self.delta[i]=0
            for j in range(0,len(nextLayer.delta)):
                self.delta[i]+=nextLayer.delta[j]*nextLayer.array[j].weight[i]*o*(1-o)
    def adjust(self,epsilon):
        for i in range(0,len(self.array)):
            self.array[i].weight-=self.delta[i]*epsilon*self.output[i]
    def setDelta(self,l):
        self.delta=l.copy()
    
class Net:
    def __init__(self,epsilon):
        self.arr=[Layer(12,4),Layer(4,1)]
        self.epsilon=epsilon
    def calc(self,inputArr):
        arr=inputArr
        for i in self.arr:
            arr=i.feedForward(arr)
        return arr[0]
    def loss(ans,calc):
        return calc-ans
    def backPropagate(self,inputArr,ans):
        cal=self.calc(inputArr)
        err=Net.loss(ans,cal)
        for i in range(1,len(self.arr)+1):
            if i==1:
                l=[]
                for j in range(0,len(self.arr.delta)):
                    l.append(err*cal*(1-cal))
                self.setDelta(l)
            else:
                self.arr[-i].calcDelta(self.arr[-(i-1)],cal)
