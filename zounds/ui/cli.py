import argparse


class BasePartialArgumentParser(argparse.ArgumentParser):
    def __init__(self, groupname, group_description):
        super(BasePartialArgumentParser, self).__init__(add_help=False)
        self.group = self.add_argument_group(groupname, group_description)

    def add_argument(self, *args, **kwargs):
        self.group.add_argument(*args, **kwargs)


class ObjectStorageSettings(BasePartialArgumentParser):
    def __init__(self):
        super(ObjectStorageSettings, self).__init__(
            'object_storage',
            'Rackspace object storage settings for model checkpoint storage')
        self.add_argument(
            '--object-storage-region',
            help='the rackspace object storage region',
            default='DFW')
        self.add_argument(
            '--object-storage-username',
            help='rackspace cloud username',
            required=True)
        self.add_argument(
            '--object-storage-api-key',
            help='rackspace cloud api key',
            required=True)


class AppSettings(BasePartialArgumentParser):
    def __init__(self):
        super(AppSettings, self).__init__(
            'app',
            'In-browser REPL settings')
        self.add_argument(
            '--app-secret',
            help='app password. If not provided, REPL is public',
            required=False)
        self.add_argument(
            '--port',
            help='The port on which the In-Browser REPL app should listen',
            default=8888)


class NeuralNetworkTrainingSettings(BasePartialArgumentParser):
    def __init__(self):
        super(NeuralNetworkTrainingSettings, self).__init__(
            'training',
            'Common settings for training neural networks')
        self.add_argument(
            '--epochs',
            help='how many passes over the data should be made during training',
            type=int)
        self.add_argument(
            '--batch-size',
            help='how many examples constitute a minibatch?',
            type=int,
            default=64)
        self.add_argument(
            '--nsamples',
            help='the number of samples to draw from the database for training',
            type=int)
