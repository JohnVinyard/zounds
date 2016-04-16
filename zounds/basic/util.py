import os


def iter_files(base_path):
    for fn in os.listdir(base_path):
        yield os.path.join(base_path, fn)


def process_dir(base_path, process_func):
    for fp in iter_files(base_path):
        process_func(fp)