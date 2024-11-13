import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

def ML(f):
    def Handle(*a, **b):
        q = f(*a, **b)
        weights = q[:-1]
        intercept = q[-1]
        return weights, intercept
    return Handle


@ML
def SVM(X, y):

    def Optimize(params, X, y):
        W = params[:-1]
        b = params[-1]
        loss = 0.5*np.dot(W, W)
        total = np.maximum(0, 1 - y*(np.dot(X, W) + b))
        return loss + np.sum(total)

        
    m, n = X.shape
    w = np.zeros(n+1)

    res = minimize(Optimize, w, method='SLSQP', args=(X, y))
    
    return res.x

def Line(X, W, b, n=40):
    x0, x1 = np.min(X[:,0]), np.max(X[:,0])
    y0, y1 = np.min(X[:,1]), np.max(X[:,1])
    dx = (x1 - x0)/(n - 1)
    dy = (y1 - y0)/(n - 1)
    qx, qy = [], []
    for i in range(n):
        h = x0 + i*dx
        g = y0 + i*dy
        qx.append(h)
        qy.append(W[0]*h + W[1]*g + b)
    return qx, qy

def Plane(X, W, b, n=40):
    x0, x1 = np.min(X[:,0]), np.max(X[:,0])
    y0, y1 = np.min(X[:,1]), np.max(X[:,1])
    dx = (x1 - x0)/(n - 1)
    dy = (y1 - y0)/(n - 1)
    planex = [x0+i*dx for i in range(n)]
    planey = [y0+i*dy for i in range(n)]

    x, y = np.meshgrid(planex, planey)
    z = []
    for i in range(n):
        tz = []
        for j in range(n):
            answer = W[0]*x[i, j] + W[1]*y[i, j] + b
            tz.append(answer)
        z.append(tz)

    z = np.array(z)
    return x, y, z
        

rows = 100

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
ay = fig.add_subplot(122, projection='3d')


X = np.random.randn(rows, 2)
y = np.array([-1 if int(i*100) % 2 == 0 else 1 for i, j in X])

W, b = SVM(X, y)

predictions = X @ W + b

lx, ly = Line(X, W, b)

for px, qx, qy in zip(predictions, X[:, 0], X[:, 1]):
    ax.scatter(qx, qy, color='red' if px < 0 else 'green')
    ay.scatter(qx, qy, px - 1 if px < 0 else px + 1, color='red' if px < 0 else 'green')


ax.plot(lx, ly, color='black')

px, py, pz = Plane(X, W, b)

ay.plot_surface(px, py, pz, color='black')

plt.show()











