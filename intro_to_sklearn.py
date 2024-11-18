# model = SVC()
# coef = model.coef_[0]
# b0 = clf.intercept_
# b1 = -coef[0]/coef[1]

import numpy as np
import matplotlib.pyplot as plt

# Import the SVM class
from sklearn.svm import SVC

# Generate classification boundary line
def Line(x, model, n=50):
    xmin, xmax = np.min(x), np.max(x)
    dx = (xmax - xmin)/(n - 1)
    lx, ly = [], []
    coef = model.coef_[0]
    b0 = model.intercept_
    b1 = -coef[0]/coef[1]
    ix = np.arange(xmin, xmax+dx, dx)
    iy = b0 + b1*ix
    return ix, iy
    
# Create sample data
rows = 300
X = np.random.randn(rows, 2)

y = np.array([-1 if int(i[0]*100) % 2 == 0 else 1 for i in X])

# Declare model
model = SVC(kernel='linear')

# Train Test Split
prop = 0.8
I = int(prop*len(y))

trainX, trainY = X[:I], y[:I]
testX = X[I:]

# Fit and predict
model.fit(trainX, trainY)
predictions = model.predict(testX)

# Plot results
fig = plt.figure()
ax = fig.add_subplot(111)

for (a, b), p in zip(testX, predictions):
    ax.scatter(a, b, color='green' if p == 1 else 'red')

lx, ly = Line(testX[:, 0], model)

ax.plot(lx, ly, color='black', linewidth=0.8)

plt.show()
