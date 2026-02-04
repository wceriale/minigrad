import torch

class Bigram:

    # generate an NxN matrix with counts of the occurences
    def __init__(self, file):
        with open(file, 'r') as f:
            lines = f.read().splitlines()
        # . represents start and end of word
        self._alphabet = list('.abcdefghijklmnopqrstuvwxyz')
        self._intMap = {c: i for i, c in enumerate(self._alphabet)}
        self._N = torch.zeros((27, 27), dtype=torch.int32)
        for word in lines:
            word = '.' + word + '.'
            for x, y in zip(word, word[1:]):
                ix1 = self._stoi(x)
                ix2 = self._stoi(y)
                self._N[ix1, ix2] += 1
        # Add 1 to smooth the model and prevent counts of 0
        self._P = (self._N+1).float() / (self._N+1).float().sum(1, keepdim=True)

    def _stoi(self, s):
        return self._intMap[s]

    def _itos(self, i):
        return self._alphabet[i]
    
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


b = Bigram('names.txt')
names = ['zoey', 'emmajq']
b.calc(names)
# names = b.makeNames(20)
# print('\n'.join(names))