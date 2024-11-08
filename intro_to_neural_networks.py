import numpy as np

# Activation Function
def sigmoid(x, derv=False):
    f = 1.0/(1.0 + np.exp(-x))
    if derv:
        return f*(1 - f)
    return f

# MinMax Normalization Class
class Normalize:

    def __init__(self, inputs, outputs):
        self.x = inputs
        self.y = outputs

        self.minx = 0
        self.maxx = 0
        self.miny = 0
        self.maxy = 0

    # Normalize data
    def norm(self):
        self.minx = np.min(self.x)
        self.maxx = np.max(self.x)
        self.miny = np.min(self.y)
        self.maxy = np.max(self.y)
        return (self.x - self.minx)/(self.maxx - self.minx), (self.y - self.miny)/(self.maxy - self.miny)

    # Convert predictions back to original scale
    def unnorm(self, predictions):
        return predictions*(self.maxy - self.miny) + self.miny


# Define inputs
inputs = [1, 2, 5, 3, 8, 4]
outputs = [3, 9, 1]

# Normalize data
X, Y = np.array(inputs), np.array(outputs)
norm = Normalize(X, Y)

nX, nY = norm.norm()

# Generate neural network objects
W1 = np.random.rand(6, 5)
L1 = np.zeros(5)
SL1 = np.zeros(5)

W2 = np.random.rand(5, 4)
L2 = np.zeros(4)
SL2 = np.zeros(4)

W3 = np.random.rand(4, 3)
L3 = np.zeros(3)
O = np.zeros(3)

epochs = 40
bias = -1

# Train the neural network
for epoch in range(epochs):
    print("Epochs: ", epoch+1)

    # Forward Propigation
    L1 = nX @ W1 + bias
    SL1 = sigmoid(L1)

    L2 = SL1 @ W2 + bias
    SL2 = sigmoid(L2)

    L3 = SL2 @ W3 + bias
    O = sigmoid(L3)

    # Back Propigation
    error = (nY - O)**2
    #print(error)

    delta = []
    delta.append(2*(nY - O)*sigmoid(L3, derv=True))
    W3 += delta[0]

    error = W3 @ delta[0]
    delta.append(error*sigmoid(L2, derv=True))
    W2 += delta[1]

    error = W2 @ delta[1]
    delta.append(error*sigmoid(L1, derv=True))
    W1 += delta[2]


Yh = norm.unnorm(O)
print(Y)
print(Yh)
print("Final Error: ", Y - Yh)
    
    
    
    
    

    








