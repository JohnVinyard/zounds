import featureflow as ff
from random_samples import ShuffledSamples
from random_samples import InfiniteSampler
from preprocess import PreprocessingPipeline


def learning_pipeline():
    class LearningPipeline(ff.BaseModel):
        samples = ff.PickleFeature(ff.IteratorNode)

        shuffled = ff.PickleFeature(
            ShuffledSamples,
            nsamples=ff.Var('nsamples'),
            dtype=ff.Var('dtype'),
            needs=samples)

    return LearningPipeline


def infinite_streaming_learning_pipeline(cls):
    roots = filter(lambda feature: feature.is_root, cls.features.itervalues())

    if len(roots) != 1:
        raise ValueError('cls must have a single root feature')

    root = roots[0]

    class InfiniteLearningPipeline(cls):
        dataset = ff.Feature(
            InfiniteSampler,
            nsamples=ff.Var('nsamples'),
            dtype=ff.Var('dtype'))

        pipeline = ff.ClobberPickleFeature(
            PreprocessingPipeline,
            needs=cls.features,
            store=True)

        @classmethod
        def load_network(cls):
            if not cls.exists():
                raise RuntimeError('No network has been trained or saved')

            instance = cls()
            for p in instance.pipeline:
                try:
                    return p.network
                except AttributeError:
                    pass

            raise RuntimeError('There is no network in the pipeline')

    root.needs = InfiniteLearningPipeline.dataset
    InfiniteLearningPipeline.__name__ = cls.__name__
    InfiniteLearningPipeline.__module__ = cls.__module__

    return InfiniteLearningPipeline
