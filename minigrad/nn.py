from minigrad.engine import Value
import random 

class Neuron:
    # nin: number of inputs into this neuron
    def __init__(self, nin):
        # define our weights/bias for given input
        self.w = [Value(random.uniform(-1, 1)) for _ in range (nin)]
        self.b = Value(random.uniform(-1, 1))
        

    def __call__(self, x):
        # w * x + b   | w * x represents the dot product between these values
        act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        out = act.tanh()
        return out
    
x = [2.0, 3.0]
n = Neuron(2)

print(n(x))
