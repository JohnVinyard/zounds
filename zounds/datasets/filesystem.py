import os


class Directory(object):
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for f in os.listdir(self.path):
            p = os.path.join(self.path, f)
            if not os.path.isdir(p):
                yield os.path.join(self.path, f)
