import featureflow as ff


def simple_lmdb_settings(path, map_size=1e9):
    """
    Creates a decorator that can be used to configure sane default LMDB
    persistence settings for a model

    :param path: The path where the LMDB database files will be created
    :param map_size: The amount of space to allot for the LMDB database
    :return: a decorator that configures persistence settings
    """
    def decorator(cls):
        class Settings(ff.PersistenceSettings):
            id_provider = ff.UuidProvider()
            key_builder = ff.StringDelimitedKeyBuilder()
            database = ff.LmdbDatabase(
                    path, key_builder=key_builder, map_size=map_size)

        class Model(cls, Settings):
            pass

        return Model

    return decorator
