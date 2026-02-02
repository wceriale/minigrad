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
    
class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs =  [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
class MLP:

    # nin: number of inputs
    # nouts: list of layer sizes [2, 3] -> Layer of 2 neurons, Layer of 3 neurons
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        # Append number of inputs to front, build all layers for the MLP
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        

    def __call__(self, x):
        input = x
        for layer in self.layers:
            # Set layer output to be the next layer input
            output = layer(input)
            input = output
        return output
    
x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])

print(n(x))
