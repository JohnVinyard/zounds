import numpy as np
import featureflow as ff

def sigmoid(a):
    return 1. / (1 + np.exp(-a))


def stochastic_binary(a):
    return a > np.random.random_sample(a.shape)


def simple_settings(cls):
    """
    Create sane default persistence settings for learning pipelines
    :param cls: The class to decorate
    """
    class Settings(ff.PersistenceSettings):
        _id = cls.__name__
        id_provider = ff.StaticIdProvider(_id)
        key_builder = ff.StringDelimitedKeyBuilder()
        database = ff.FileSystemDatabase(
                path=_id, key_builder=key_builder, createdirs=True)

    class Model(cls, Settings):
        pass

    return Model

