class Trainer(object):
    def __init__(self, epochs, batch_size, checkpoint_epochs=1):
        super(Trainer, self).__init__()
        self.checkpoint_epochs = checkpoint_epochs
        self.batch_size = batch_size
        self.epochs = epochs
        self._batch_complete_callbacks = dict()
        self._current_epoch = 0

    def train(self):
        raise NotImplemented()

    def register_batch_complete_callback(self, callback):
        self._batch_complete_callbacks[id(callback)] = callback

    def unregister_batch_complete_callback(self, callback):
        try:
            del self._batch_complete_callbacks[id(callback)]
        except KeyError:
            # the callback was never registered
            pass

    def on_batch_complete(self, *args, **kwargs):
        for callback in self._batch_complete_callbacks.itervalues():
            callback(*args, **kwargs)

