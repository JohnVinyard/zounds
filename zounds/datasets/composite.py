

class CompositeDataset(object):
    """
    A dataset composed of two or more others

    Args:
        datasets (list of datasets): One or more other datasets

    Examples:
        >>> from zounds import InternetArchive, CompositeDataset, ingest
        >>> dataset1 = InternetArchive('beethoven_ingigong_850')
        >>> dataset2 = InternetArchive('The_Four_Seasons_Vivaldi-10361')
        >>> composite = CompositeDataset(dataset1, dataset2)
        >>> ingest(composite, Sound) # ingest data from both datasets
    """

    def __init__(self, *datasets):
        super(CompositeDataset, self).__init__()
        self.datasets = datasets

    def __iter__(self):
        for dataset in self.datasets:
            for meta in dataset:
                yield meta
