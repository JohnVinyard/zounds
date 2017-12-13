import numpy as np
import featureflow as ff
from random_samples import ShuffledSamples


def learning_pipeline(dtype=np.float32):
    class LearningPipeline(ff.BaseModel):
        samples = ff.PickleFeature(ff.IteratorNode)

        shuffled = ff.PickleFeature(
            ShuffledSamples,
            nsamples=ff.Var('nsamples'),
            dtype=dtype,
            needs=samples)

    return LearningPipeline
