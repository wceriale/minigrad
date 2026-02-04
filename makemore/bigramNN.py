import torch
import torch.nn.functional as F

class BigramNN:

    # generate an NxN matrix with counts of the occurences
    # when count is 0, do all
    def __init__(self, file, count=0):
        with open(file, 'r') as f:
            lines = f.read().splitlines()
        # . represents start and end of word
        self._alphabet = list('.abcdefghijklmnopqrstuvwxyz')
        self._intMap = {c: i for i, c in enumerate(self._alphabet)}
        # Input is a onehot of the alphabet (with ending/starting char). So each char is a vector of 27. 
        # Output is all possible letters w/ probability. Also 27. 
        g = torch.Generator().manual_seed(2147483647)
        self._W = torch.randn((27, 27), generator=g, requires_grad=True)
        self._xs = []
        self._ys = []
        if count == 0:
            for word in lines:
                word = '.' + word + '.'
                for x, y in zip (word, word[1:]):
                    self._xs.append(self._stoi(x))
                    self._ys.append(self._stoi(y))
        else:
            num_lines = min(count, len(lines))
            for i in range(num_lines):
                word = '.' + lines[i] + '.'
                for x, y in zip (word, word[1:]):
                    self._xs.append(self._stoi(x))
                    self._ys.append(self._stoi(y))

        self._xs = torch.tensor(self._xs)
        self._ys = torch.tensor(self._ys)

    def _stoi(self, s):
        return self._intMap[s]

    def _itos(self, i):
        return self._alphabet[i]
    

    def SGD(self, update_amount, count):
        x_enc = F.one_hot(torch.tensor(self._xs), num_classes=27).float()

        for i in range(count):
            logits = x_enc @ self._W # log count
            # softmax
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdim=True)

            # get loss
            loss = -probs[torch.arange(self._xs.nelement()), self._ys].log().mean()

            print('loss at step i:{i} =' + str(loss.item()))
            self._W.grad = None  
            loss.backward()      

            # Update
            self._W.data += -update_amount * self._W.grad

        print('loss at the very end=' + str(loss.item()))


    def calc(self, words):
        log_likelihood = 0
        n = 0
        for word in words:
            chs = '.' + word + '.'
            for x, y in zip(chs, chs[1:]):
                ix1 = self._stoi(x)
                ix2 = self._stoi(y)
                prob = self._P[ix1, ix2]
                logprob = torch.log(prob)
                log_likelihood += logprob
                n += 1
                print(f'ch1={x}, ch2={y}, prob={prob: .4f}, logprob={logprob: 0.4f}')
            print('-ll=' + str(-log_likelihood / n))


    # Create n names
    def makeNames(self, n):
        names = []
        g = torch.Generator().manual_seed(2147483647)
        for _ in range(n):
            n = 0
            name = ''
            index = 0
            while True:
                nextIndex = torch.multinomial(self._P[index], num_samples=1, replacement=True, generator=g).item()
                letter = self._itos(nextIndex)

                if nextIndex == 0: 
                    break
                name += letter
                index = nextIndex
            names.append(name)
        return names


b = BigramNN('names.txt')
b.SGD(50, 100)