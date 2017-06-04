from preprocess import Preprocessor, PreprocessResult, Op
import hashlib
import os


class KerasPreprocessResult(PreprocessResult):
    def __init__(
            self, path, data, op, inversion_data=None, inverse=None, name=None):
        super(KerasPreprocessResult, self).__init__(
            data, op, inversion_data, inverse, name)
        self.path = path

    def __getstate__(self):
        encoder = self.op.model
        encoder_filename = 'encoder.{hash}.h5'.format(
            hash=hashlib.md5(encoder.to_json()).hexdigest())
        encoder_path = os.path.join(self.path, encoder_filename)
        encoder.save(encoder_path)
        self.op._kwargs['model'] = encoder_path

        decoder = self.inversion_data.model
        decoder_filename = 'decoder.{hash}.h5'.format(
            hash=hashlib.md5(decoder.to_json()).hexdigest())
        decoder_path = os.path.join(self.path, decoder_filename)
        decoder.save(decoder_path)
        self.inversion_data._kwargs['model'] = decoder_path
        d = self.__dict__
        for k, v in d.iteritems():
            try:
                d[k] = v.__dict__
            except AttributeError:
                pass
        return d

    def __setstate__(self, state):
        from keras.models import load_model
        inversion_data = state['inversion_data']
        inversion_data['_kwargs']['model'] = load_model(
            inversion_data['_kwargs']['model'])
        op = state['op']
        op['_kwargs']['model'] = load_model(
            op['_kwargs']['model'])

        self.name = state['name']
        self.inverse = Op(
            state['inverse']['_func'],
            **state['inverse']['_kwargs'])
        self.inversion_data = Op(
            inversion_data['_func'],
            **inversion_data['_kwargs'])
        self.data = state['data']
        self.op = Op(op['_func'], **op['_kwargs'])

    def for_storage(self):
        return KerasPreprocessResult(
            self.path,
            None,
            self.op,
            self.inversion_data,
            self.inverse,
            self.name)


class KerasModel(Preprocessor):
    def __init__(
            self,
            architecture_func=None,
            path=None,
            optimizer='adadelta',
            loss='mean_absolute_error',
            epochs=100,
            batch_size=256,
            needs=None):
        super(KerasModel, self).__init__(needs=needs)
        self.path = path
        self.architecture_func = architecture_func
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer

    def _forward_func(self):
        def x(d, model=None):
            return model.predict(d)

        return x

    def _backward_func(self):
        def x(d, model=None):
            return model.predict(d)

        return x

    def _process(self, data):
        data = self._extract_data(data)
        trainable, encoder, decoder = self.architecture_func(data)
        trainable.compile(optimizer=self.optimizer, loss=self.loss)
        trainable.fit(
            data, data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(data, data))

        processed_data = encoder.predict(data)
        hashed = hashlib.md5(trainable.to_json()).hexdigest()

        op = self.transform(model=encoder)
        inv_data = self.inversion_data(model=decoder)
        inv = self.inverse_transform()

        yield KerasPreprocessResult(
            self.path,
            processed_data,
            op,
            inversion_data=inv_data,
            inverse=inv,
            name='KerasModel.{hashed}'.format(**locals()))