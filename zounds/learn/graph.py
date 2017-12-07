import numpy as np
import featureflow as ff
from random_samples import ShuffledSamples


def learning_pipeline(nsamples=int(1e5), dtype=np.float32):
    class LearningPipeline(ff.BaseModel):
        samples = ff.PickleFeature(ff.IteratorNode)

        shuffled = ff.PickleFeature(
            ShuffledSamples,
            nsamples=nsamples,
            dtype=dtype,
            needs=samples)

    return LearningPipeline
