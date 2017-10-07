from multiprocessing.pool import ThreadPool, Pool, cpu_count
from itertools import repeat, izip


def ingest_one(arg):
    metadata, cls, skip_if_exists = arg
    request = metadata.request
    url = request.url
    if skip_if_exists and cls.exists(request.url):
        print 'already processed {request.url}'.format(**locals())
        return

    try:
        print 'processing {request.url}'.format(**locals())
        cls.process(meta=metadata, _id=url)
    except Exception as e:
        print e


def ingest(
        dataset,
        cls,
        skip_if_exists=True,
        multi_process=False,
        multi_threaded=False,
        cores=None):

    if multi_process:
        pool = Pool(cores or cpu_count())
        map_func = pool.map
    elif multi_threaded:
        pool = ThreadPool(cores or cpu_count())
        map_func = pool.map
    else:
        map_func = map

    cls_args = repeat(cls)
    skip_args = repeat(skip_if_exists)

    map_func(ingest_one, izip(dataset, cls_args, skip_args))
