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
    
    def parameters(self):
        return self.w + [self.b] # append the bias as an additional weight value
    
class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs =  [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    # Get all the parameters for this layer
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
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
    
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()] 
