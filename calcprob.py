'''
calcprob.py, for calculating the probability.
'''





import collections



class Calcprob:
    def __init__(self, beta, sample_min, max_len=3000):
        assert 0. <= sample_min <= 1., "Invalid sample_min"
        assert beta > 0, "Invalid beta"
        self.beta = beta
        self.sample_min = sample_min
        self.max_len = max_len
        self.history = collections.deque(maxlen=max_len)
        self.num_slot = 1000
        self.hist = [0] * self.num_slot
        self.count = 0

    def update_history(self, losses):
        """
        BoundedHistogram
        :param losses:
        :return:
        """
        for loss in losses:
            assert loss > 0
            if self.count is self.max_len:
                loss_old = self.history.popleft()
                slot_old = int(loss_old * self.num_slot) % self.num_slot
                self.hist[slot_old] -= 1
            else:
                self.count += 1
                self.history.append(loss)
            slot = int(loss * self.num_slot) % self.num_slot
            self.hist[slot] += 1

    def get_probability(self, loss):
        assert loss > 0
        slot = int(loss * self.num_slot) % self.num_slot
        prob = sum(self.hist[:slot]) / self.count
        assert isinstance(prob, float), "int division error..."
        return prob ** self.beta

    def calc_probability(self, losses):
        if isinstance(losses, float):
            losses =  (losses, )
        self.update_history(losses)
        probs = (
            max(
                self.get_probability(loss),
                self.sample_min
            )
            for loss in losses
        )
        return probs

    def __call__(self, losses):
        return self.calc_probability(losses)





if __name__ == "__main__":
    pass






