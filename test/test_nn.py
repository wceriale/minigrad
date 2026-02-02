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
# print(ypred)

# loss_arr = [(yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)]

# loss.backward()


for k in range(20):
    # forward pass
    ypred = [n(x) for x in xs]
    loss_arr = [(yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)]
    loss = sum(loss_arr)

    # backprop
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.05 * p.grad

    # loss value given k iteration
    print(k, loss.data, ypred)
