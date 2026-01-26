import math


class Value: 

    # Children represents the Values used to create this Value.
    # Visually, these would be the leaf nodes of the current node.
    # Parent node should explicitly set self.grad to 1.
    def __init__(self, data, _children =(), _op='', name=''):
        self.data = data
        self._children = set(_children)
        self._op = _op
        self.name = name

        # Start at 0.0. The input does not effect the output.
        # Grad for a given Value x and it's "parent" Value y, is dy/dx
        self.grad = 0.0

        # Function that defines backprop. 
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        # add only copies the gradient from the parent into the children.
        def _backward():
            self.grad = out.grad
            other.grad = out.grad

        out._backward = _backward
        return out
    
    def __mult__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        # mult is the value of neighbor * output gradient.
        def _backward():
            self.grad = out.grad * other.data
            other.grad = out.grad * self.data

        out._backward = _backward
        return out
    
    def tanh(self):
        t = (math.exp(2*self.data) - 1) / (math.exp(2*self.data) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad = (1 - t ** 2) * out.grad

        out._backward = _backward
        return out