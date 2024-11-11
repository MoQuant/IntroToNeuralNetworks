import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Neural Network Class
class NeuralNetwork:

    def __init__(self, iN, oN, epochs=500, lr=1.0):
        self.M = iN
        self.N = oN
        self.epochs = list(range(epochs))
        self.lr = lr

    # Trains data
    def __call__(self, x, y):
        bias = -1
        for epoch in self.epochs:
            # Forward propigation
            for i in self.axis:
                if i == self.axis[0]:
                    self.L[i] = x @ self.W[i] + bias
                else:
                    self.L[i] = self.SL[i+1] @ self.W[i] + bias
                self.SL[i] = self.sigmoid(self.L[i])

            # Back Propigation
            delta = {}
            for i in self.baxis:
                if i == self.baxis[0]:
                    error = (y - self.SL[i])**2
                    delta[i] = 2.0*(y - self.SL[i])*self.sigmoid(self.L[i], derv=True)
                else:
                    error = self.W[i-1] @ delta[i-1]
                    delta[i] = error*self.sigmoid(self.L[i], derv=True)

            for i in self.axis:
                self.W[i] += self.lr*delta[i]

    # Test output
    def testData(self, x):
        bias = -1
        for i in self.axis:
            if i == self.axis[0]:
                self.L[i] = x @ self.W[i] + bias
            else:
                self.L[i] = self.SL[i+1] @ self.W[i] + bias
            self.SL[i] = self.sigmoid(self.L[i])
        return self.SL[self.axis[-1]]

    # Activation and optimization function
    def sigmoid(self, x, derv=False):
        f = 1.0 / (1.0 + np.exp(-x))
        if derv:
            return f*(1 - f)
        return f

    # Builds parameters
    def buildWeights(self):
        self.axis = list(range(self.M, self.N, -1))
        self.baxis = self.axis[::-1]

        self.W = {i:np.random.rand(i, i-1) for i in self.axis}
        self.L = {i:np.zeros(i-1) for i in self.axis}
        self.SL = {i:np.zeros(i-1) for i in self.axis}

    # Split data into training and testing
    def trainTestSplit(self, x, y, proportion=0.7):
        I = int(proportion*len(x))
        trainX = x[:I]
        trainY = y[:I]
        testX = x[I:]
        testY = y[I:]
        return trainX, trainY, testX, testY
    
    # Normalize the data via min/max method
    def normalizeData(self, x, y):
        x, y = np.array(x), np.array(y)
        self.lowx = {}
        self.highx = {}
        self.lowy = {}
        self.highy = {}
        m, n = self.M, self.N
        for i in range(m):
            self.lowx[i] = np.min(x[:, i])
            self.highx[i] = np.max(x[:, i])
            x[:, i] = (x[:, i] - self.lowx[i])/(self.highx[i] - self.lowx[i])
        for i in range(n):
            self.lowy[i] = np.min(y[:, i])
            self.highy[i] = np.max(y[:, i])
            y[:, i] = (y[:, i] - self.lowy[i])/(self.highy[i] - self.lowy[i])
        return x, y
    
    # Convert normalized data back to original scale
    def denomalizeData(self, y):
        y = np.array(y)
        m, n = y.shape
        for i in range(n):
            y[:, i] = y[:, i]*(self.highy[i] - self.lowy[i]) + self.lowy[i]
        return y

        

rows = 100
X = 100*np.random.rand(rows, 15)
Y = 10*np.random.rand(rows, 3)

EX, EY, EZ = [], [], []
for epoch in (50, 100, 150, 200, 250, 300):
    Ex, Ey, Ez = [], [], []
    for lr in (0.01, 0.05, 0.1, 0.5, 1.0, 2.0):
        AI = NeuralNetwork(15, 3, epochs=epoch, lr=lr)
        AI.buildWeights()

        trainX, trainY, testX, testY = AI.trainTestSplit(X, Y)

        nTrainX, nTrainY = AI.normalizeData(trainX.tolist(), trainY.tolist())
        nTestX, nTestY = AI.normalizeData(testX.tolist(), testY.tolist())

        for inx, iny in zip(nTrainX, nTrainY):
            AI(inx, iny)

        Ypred = []
        for inx in nTestX:
            iny = AI.testData(inx).tolist()
            Ypred.append(iny)
            
        YHat = AI.denomalizeData(np.array(Ypred))

        RMSE = 0
        for actual, prediction in zip(testY, YHat):
            RMSE += np.sum(pow(actual - prediction, 2))

        RMSE = np.sqrt(RMSE/(len(YHat) - 1))
        print(epoch, lr)

        Ex.append(epoch); Ey.append(lr); Ez.append(RMSE)

    EX.append(Ex); EY.append(Ey); EZ.append(Ez)

EX, EY, EZ = [np.array(u) for u in (EX, EY, EZ)]

ax.set_title("Root Mean Squared Error")
ax.set_xlabel("Epochs")
ax.set_ylabel("Learning Rate")
ax.set_zlabel("RMSE")

ax.plot_surface(EX, EY, EZ, cmap='viridis', edgecolor='black', linewidth=0.7)
plt.show()








