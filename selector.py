'''
selector.py, for calculating the probability and backward the gradient.
'''







import calcprob
import numpy as np


class Selector:

    def __init__(self, batch_size,
                 beta, sample_min, max_len=3000):
        self.batch_size = batch_size
        self.calcprob = calcprob.Calcprob(beta,
                                          sample_min,
                                          max_len)
        self.reset()

    def backward(self):
        loss = sum(self.batch)
        loss.backward()
        self.reset()

    def reset(self):
        self.batch = []
        self.length = 0.

    def select(self, losses):
        probs = self.calcprob(losses)
        for i, prob in enumerate(probs):
            if np.random.rand() < prob:
                self.batch.append(losses[i])
                self.length += 1
                if self.length >= self.batch_size:
                    self.backward()

    def __call__(self, losses):
        self.select(losses)








