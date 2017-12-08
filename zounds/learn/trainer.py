class Trainer(object):
    def __init__(self, epochs, batch_size):
        super(Trainer, self).__init__()
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self):
        raise NotImplemented()
