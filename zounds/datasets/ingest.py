from multiprocessing.pool import ThreadPool, Pool, cpu_count
from itertools import repeat, izip, imap


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

    pool = None

    if multi_process:
        pool = Pool(cores or cpu_count())
        map_func = pool.imap_unordered
    elif multi_threaded:
        pool = ThreadPool(cores or cpu_count())
        map_func = pool.imap_unordered
    else:
        map_func = map

    cls_args = repeat(cls)
    skip_args = repeat(skip_if_exists)

    map_func(ingest_one, izip(dataset, cls_args, skip_args))

    if pool is not None:
        # if we're ingesting using multiple processes or threads, the processing
        # should be parallel, but this method should be synchronous from the
        # caller's perspective
        pool.close()
        pool.join()

