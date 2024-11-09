import numpy as np


class NNET:

    def __init__(self, inx, outx, epochs=100):
        self.m = inx
        self.n = outx
        self.epochs = range(epochs)

        self.minx = {}
        self.maxx = {}
        self.miny = {}
        self.maxy = {}

    def __call__(self, x, y):
        bias = -1
        for epoch in self.epochs:
            # Forward propigation
            for step in self.axis:
                if step == self.axis[0]:
                    self.L[step] = x @ self.W[step] + bias
                    self.SL[step] = self.sigmoid(self.L[step])
                else:
                    self.L[step] = self.SL[step+1] @ self.W[step] + bias
                    self.SL[step] = self.sigmoid(self.L[step])

            # Backpropigation
            update = {}
            for step in self.raxis:
                if step == self.raxis[0]:
                    error = (y - self.SL[step])**2
                    delta = 2*(y - self.SL[step])*self.sigmoid(self.L[step], derv=True)
                    update[step] = delta
                else:
                    error = self.W[step-1] @ delta
                    delta = error*self.sigmoid(self.L[step], derv=True)
                    update[step] = delta

            for i in self.axis:
                self.W[i] += update[i]

    def testmodel(self, x):
        bias = -1
        for step in self.axis:
            if step == self.axis[0]:
                self.L[step] = x @ self.W[step] + bias
                self.SL[step] = self.sigmoid(self.L[step])
            else:
                self.L[step] = self.SL[step+1] @ self.W[step] + bias
                self.SL[step] = self.sigmoid(self.L[step])
        return self.SL[self.axis[-1]]

    def normalization(self, x, y):
        m, n = x.shape
        for i in range(n):
            self.minx[i] = min(x[:, i])
            self.maxx[i] = max(x[:, i])
        o, p = y.shape
        for i in range(p):
            self.miny[i] = min(y[:, i])
            self.maxy[i] = max(y[:, i])
        for i in range(n):
            x[:, i] = (x[:, i] - self.minx[i])/(self.maxx[i] - self.minx[i])
        for i in range(p):
            y[:, i] = (y[:, i] - self.miny[i])/(self.maxy[i] - self.miny[i])
        return x, y

    def convert(self, y):
        o, p = y.shape
        for i in range(p):
            y[:, i] = y[:, i]*(self.maxy[i] - self.miny[i]) + self.miny[i]
        return y

    def build(self):
        self.axis = list(range(self.m, self.n, -1))
        self.raxis = self.axis[::-1]

        self.W = {}
        self.L = {}
        self.SL = {}

        for i in self.axis:
            self.W[i] = np.random.rand(i, i-1)
            self.L[i] = np.zeros(i-1)
            self.SL[i] = np.zeros(i-1)

    def sigmoid(self, x, derv=False):
        f = 1.0 / (1.0 + np.exp(-x))
        if derv:
            return f*(1 - f)
        return f

    def splitter(self, X, Y, prop=0.7):
        n = int(prop*len(X))
        tX = X[:n]
        tY = Y[:n]
        eX = X[n:]
        eY = Y[n:]
        return tX, tY, eX, eY
        


net = NNET(6, 2)

X = (100*np.random.rand(50, 6))
Y = (100*np.random.rand(50, 2))

nX, nY = net.normalization(X, Y)

trainX, trainY, testX, testY = net.splitter(nX, nY)

net.build()

for ix, iy in zip(trainX, trainY):
    net(ix, iy)

yhat = net.convert(net.testmodel(testX))
testY = net.convert(testY)

for y0, yh in zip(testY, yhat):
    print(y0, yh, y0 - yh)
    

