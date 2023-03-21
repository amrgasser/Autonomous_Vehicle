class LossHistory:
    def __init__(self, smoothing_factor=0.0):
        self.alpha = smoothing_factor
        self.loss = []

    def append(self, value):
        self.loss.append(
            self.alpha*self.loss[-1] + (1-self.alpha)*value if len(self.loss) > 0 else value)

    def get(self):
        return self.loss
