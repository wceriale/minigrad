import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class BigramMLP:
    # generate an NxN matrix with counts of the occurences
    # when count is 0, do all
    def __init__(self, file, context_size, count=0):
        with open(file, 'r') as f:
            lines = f.read().splitlines()
        if count != 0:
            lines = lines[:count]
        # . represents start and end of word
        self._alphabet = list('.abcdefghijklmnopqrstuvwxyz')
        self._intMap = {c: i for i, c in enumerate(self._alphabet)}
        # Input is a onehot of the alphabet (with ending/starting char). So each char is a vector of 27. 
        # Output is all possible letters w/ probability. Also 27. 
        g = torch.Generator().manual_seed(2147483647)

        self._context_size = context_size
        context = [0] * context_size
        xs = []
        ys = []
        for word in lines:
            word = word + '.'
            for ch in word:
                ix = self._intMap[ch]
                xs.append(context)
                ys.append(ix)
                context = context[1:] + [ix]
        self._xs = torch.tensor(xs)
        self._ys = torch.tensor(ys)



        # We are going to map each letter into 2D space
        self._C = torch.randn(27, 12, generator=g, requires_grad=True)
    
        # emb is n rows -> x context chars -> 2D space representation of char
        self._emb = self._C[self._xs]

        self._W1 = torch.randn(36, 300, generator=g, requires_grad=True)
        self._b1 = torch.randn(300, generator=g, requires_grad=True)
        self._W2 = torch.randn(300, 27, generator=g, requires_grad=True)
        self._b2 = torch.randn(27, generator=g, requires_grad=True)

    def params(self):
        return [self._C, self._W1, self._b1, self._W2, self._b2]

    def _stoi(self, s):
        return self._intMap[s]

    def _itos(self, i):
        return self._alphabet[i]
    

    def SGD_mini(self, update_amount, count):
        # lre = torch.linspace(-3, 0, count)
        # lrs = 10 ** lre
        # lri = []
        # lossi = []
        for i in range(count):

            # Create a mini batch of only 64 elements to check against.
            ix = torch.randint(0, self._xs.shape[0], (128, )) 

            mini_emb = self._C[self._xs[ix]] # (128, 3, 12)

            h = torch.tanh(mini_emb.view(mini_emb.shape[0], 36) @ self._W1 + self._b1) # (128, 300)
            logits = h @ self._W2 + self._b2 # (128, 27)

            # get loss
            loss = F.cross_entropy(logits, self._ys[ix])

            # print(f'loss at step i:{i} = {loss.item()}')
            self._W1.grad = None
            self._W2.grad = None
            self._b1.grad = None
            self._b2.grad = None
            self._C.grad = None

            # Perform back prop
            loss.backward()   

            if i > 20000:
                update_amount = 0.01   

            # Update
            # lr = lrs[i]
            self._W1.data += -update_amount * self._W1.grad
            self._W2.data += -update_amount * self._W2.grad
            self._b1.data += -update_amount * self._b1.grad
            self._b2.data += -update_amount * self._b2.grad
            self._C.data += -update_amount * self._C.grad

            # track stats
            # lri.append(lre[i])
            # lossi.append(loss.item())

        print('mini loss at the very end=' + str(loss.item()))
        # plt.plot(lri, lossi)
        # plt.show()

    def print_loss(self):

        mini_emb = self._C[self._xs] # (total_p, 3, 10)

        h = torch.tanh(mini_emb.view(mini_emb.shape[0], 36) @ self._W1 + self._b1) # (64, 100)
        logits = h @ self._W2 + self._b2 # (64, 27)
        # get loss
        loss = F.cross_entropy(logits, self._ys)

        print('total loss=' + str(loss.item()))

    # Create n names
    def makeNames(self, n):
        names = []
        g = torch.Generator().manual_seed(2147483647)
        for _ in range(n):
            name = ''
            curr_context = [0] * self._context_size
            while True:

                # Get the current embedding for the given context length
                emb = self._C[torch.tensor([curr_context])] # 1, context_size, 2d

                # run it through the NN
                h = torch.tanh(emb.view(emb.shape[0], 36) @ self._W1 + self._b1)
                logits = h @ self._W2 + self._b2
                counts = logits.exp()
                probs = counts / counts.sum(1, keepdim=True)

                # Sample for the next index letter
                nextIndex = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
                letter = self._itos(nextIndex)

                # End if terminal letter, otherwise add it and update the context.
                if nextIndex == 0: 
                    break
                name += letter
                curr_context = curr_context[1:] + [nextIndex]

            names.append(name)
        return names


b = BigramMLP('names.txt', 3, 0)

print(f"number of parameters={sum(p.nelement() for p in b.params())}")
# b = BigramMLP('names.txt', 5, 1)
# b.SGD(1, 100)
b.SGD_mini(0.1, 50000)
b.print_loss()
print('\n'.join(b.makeNames(10)))