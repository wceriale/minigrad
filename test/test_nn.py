from minigrad.nn import MLP

n = MLP(3, [4, 4, 1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0] # desired targets
ypred = [n(x) for x in xs]
print(ypred)

loss_arr = [(yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)]
print('losses:')
print(loss_arr)

loss = sum(loss_arr)
print('final loss=' )
print(loss)

loss.backward()
print(loss.grad)

print(n.layers[1].neurons[0].w[0].grad)

print(n.layers[0].neurons[0].w[1].grad)
