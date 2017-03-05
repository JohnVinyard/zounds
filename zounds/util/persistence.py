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


def simple_object_storage_settings(container, region, username, api_key):
    def decorator(cls):
        class Settings(ff.PersistenceSettings):
            id_provider = ff.UuidProvider()
            key_builder = ff.StringDelimitedKeyBuilder()
            database = ff.ObjectStoreDatabase(
                container, username, api_key, region, key_builder=key_builder)

        class Model(cls, Settings):
            pass

        return Model

    return decorator


def simple_in_memory_settings(cls):
    """
    Decorator that returns a class that "persists" data in-memory.  Mostly
     useful for testing
    :param cls: the class whose features should be persisted in-memory
    :return: A new class that will persist features in memory
    """

    class Settings(ff.PersistenceSettings):
        id_provider = ff.UuidProvider()
        key_builder = ff.StringDelimitedKeyBuilder()
        database = ff.InMemoryDatabase(key_builder=key_builder)

    class Model(cls, Settings):
        pass

    return Model
