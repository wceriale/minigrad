import torch
import torch.nn.functional as F

class BigramNN:

    # generate an NxN matrix with counts of the occurences
    def __init__(self, file):
        with open(file, 'r') as f:
            lines = f.read().splitlines()
        # . represents start and end of word
        self._alphabet = list('.abcdefghijklmnopqrstuvwxyz')
        self._intMap = {c: i for i, c in enumerate(self._alphabet)}
        # Input is a onehot of the alphabet (with ending/starting char). So each char is a vector of 27. 
        # Output is all possible letters w/ probability. Also 27. 
        g = torch.Generator().manual_seed(2147483647)
        self._W = torch.randn((27, 27), generator=g) #, dtype=torch.int32)
        # word = '.' + lines[0] + '.'
        self._xs = []
        self._ys = []
        for word in lines:
            word = '.' + lines[0] + '.'
            for x, y in zip (word, word[1:]):
                self._xs.append(self._stoi(x))
                self._ys.append(self._stoi(y))
            break
        # print(xs)
        # print(ys)
            # x_enc = F.one_hot(xs, num_classes=27).float()
        # for word in lines:
        #     word = '.' + word + '.'
        #     for x, y in zip(word, word[1:]):
        #         ix1 = self._stoi(x)
        #         ix2 = self._stoi(y)
        #         self._N[ix1, ix2] += 1

        # Add 1 to smooth the model and prevent counts of 0
        # self._P = (self._N+1).float() / (self._N+1).float().sum(1, keepdim=True)

    def _stoi(self, s):
        return self._intMap[s]

    def _itos(self, i):
        return self._alphabet[i]
    
    def calcNN(self, words):
        x_enc = F.one_hot(torch.tensor(self._xs), num_classes=27).float()
        logits = x_enc @ self._W # log count
        
        counts = logits.exp()
        prob = counts / counts.sum(1, keepdim=True)
        print('prob shape=' + str(prob.shape))
        print('sum of 1st row:' + str(prob[0].sum()))
        print('sum of 2nd row:' + str(prob[1].sum()))
        # print(prob)

        log_likelihood = 0
        n = 0
        for word in words:
            chs = '.' + word + '.'
            ll_arr = len(word) + 1
            for i in range(len(word) + 1):
                # ix1 = chs[i]
                # ix2 = chs[i + 1]
                letter = chs[i]
                next_letter = chs[i + 1]
                next_letter_index = self._stoi(next_letter)
                # probabilities of next letter :
                print(f'curr_letter={letter} next_letter={next_letter} prob of next letter={prob[i][next_letter_index]: 0.3f}')
                

            # for x, y in zip(chs, chs[1:]):
            #     ix1 = self._stoi(x)
            #     ix2 = self._stoi(y)
            #     print(R[ix1][ix2])
            #     prob = self._P[ix1, ix2]
            #     logprob = torch.log(prob)
            #     log_likelihood += logprob
            #     n += 1
            #     print(f'ch1={x}, ch2={y}, prob={prob: .4f}, logprob={logprob: 0.4f}')
            # print('-ll=' + str(-log_likelihood / n))


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
b.calcNN(['emma'])
# names = ['zoey', 'emmajq']
# b.calc(names)
# names = b.makeNames(20)
# print('\n'.join(names))