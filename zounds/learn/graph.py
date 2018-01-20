import featureflow as ff
from random_samples import ShuffledSamples


def learning_pipeline():
    class LearningPipeline(ff.BaseModel):
        samples = ff.PickleFeature(ff.IteratorNode)

        shuffled = ff.PickleFeature(
            ShuffledSamples,
            nsamples=ff.Var('nsamples'),
            dtype=ff.Var('dtype'),
            needs=samples)

    return LearningPipeline
