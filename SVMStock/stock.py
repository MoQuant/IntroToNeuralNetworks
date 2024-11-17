import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def TrainTest(prop):
    def Convert(f):
        def Solve(*a, **b):
            X, y = f(*a, **b)
            I = int(prop*len(X))
            trainX, trainY = X[:I], y[:I]
            testX, testY = X[I:], y[I:]
            return trainX, trainY, testX, testY
        return Solve
    return Convert
            

@TrainTest(0.85)
def Technology(close, vol, window=100, output=20):
    X = []
    y = []
    for i in range(window, len(close)-output+1):
        hold = close[i-window:i]
        size = vol[i-window:i]
        ror = hold[1:]/hold[:-1] - 1.0
        cror = np.prod([(1 + r) for r in ror]) - 1.0

        price = close[i]
        ma = np.mean(hold)
        sd = np.std(hold)
        lowBB = price - 2.0*sd
        upBB = price + 2.0*sd

        vwap = np.dot(hold, size)/np.sum(size)

        X.append([price, ma, lowBB, upBB, vwap])
        y.append(0 if cror > 0 else 1)

    return np.array(X), np.array(y)
        

class Norm:

    def __init__(self, X):
        self.x = X
        m, n = X.shape
        self.n = n

    def normalize(self):
        X = self.x
        for i in range(self.n):
            H = self.x[:, i]
            mu = np.mean(H)
            sd = np.std(H)
            X[:, i] = (self.x[:, i] - mu)/sd
        return X


data = pd.read_csv("SPY.csv")[::-1]

close = data['adjClose'].values
volume = data['volume'].values

trainX, trainY, testX, testY = Technology(close, volume, window=200, output=50)

normal_train = Norm(trainX)
nX = normal_train.normalize()

normal_test = Norm(testX)
nTX = normal_test.normalize()

model = SVC(kernel='rbf', probability=True)

model.fit(nX, trainY)

pred = model.predict(testX)
prob = model.predict_proba(testX)


trade = list(map(lambda a: 'Buy' if a == 0 else 'Sell', pred))

plot_close = close[-len(trade):]

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)

ux = list(range(len(trade)))

ax.plot(ux, plot_close, color='red')
for ix, iy, name in zip(ux, plot_close, trade):
    ax.annotate(name, xy=(ix, iy))

plt.show()








