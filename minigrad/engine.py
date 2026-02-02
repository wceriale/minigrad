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
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        # add only copies the gradient from the parent into the children.
        # use += if the node is used more than once. This 'accumulates' the gradients.
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out
    
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        # mult is the value of neighbor * output gradient.
        # use += if the node is used more than once. This 'accumulates' the gradients.
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out
    
    def tanh(self):
        t = (math.exp(2*self.data) - 1) / (math.exp(2*self.data) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            # use += if the node is used more than once. This 'accumulates' the gradients.
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out
    
    def exp(self):
        t = math.exp(self)
        out = Value(t, (self, ), 'exp')

        def _backward():
            # use += if the node is used more than once. This 'accumulates' the gradients.
            self.grad += out.grad * self.data

        out._backward = _backward
        return out
    
    def __pow__(self, n):
        assert isinstance(n, (int, float)), "only support int/float"
        t = self.data**n
        out = Value(t, (self, ), 'pow')

        def _backward():
            # use += if the node is used more than once. This 'accumulates' the gradients.
            self.grad += n * self.data ** (n-1) * out.grad

        out._backward = _backward
        return out


    # Right hand value definitions for our Library class
    def __rmul__(self, other): # other * self
        return self * other
    def __radd__(self, other): # other + self
        return self + other
    def __truediv__(self, other): # self / other
        return self * other**-1
    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        return self + (-other)
    
    # Apply backprop to calculate gradients
    def backward(self):
        visited = set()
        topo = []

        # Build topological order recursively. Process children first, then yourself.
        def build_topo(n):
            if n not in visited:
                visited.add(n)
                for child in n._children:
                    build_topo(child)
                topo.append(n)

        # Build topological order with this Value as the last one appended.
        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
         
        