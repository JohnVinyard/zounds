from featureflow import Node
import numpy as np
from scipy.spatial.distance import cdist
from zounds.nputil import safe_unit_norm, sliding_window
from zounds.timeseries import ConstantRateTimeSeries


class TemplateMatch(Node):
    def __init__(self, templates=None, needs=None):
        super(TemplateMatch, self).__init__(needs=needs)
        self._windowsize = templates.shape[1:]
        twod_templates = templates.reshape(templates.shape[0], -1)
        self._templates = safe_unit_norm(twod_templates)

    def _process(self, data):
        out = np.zeros((data.shape[0], self._templates.shape[0]))
        for i, chunk in enumerate(data):
            patches = sliding_window( \
                    chunk,
                    self._windowsize,
                    (1,) * len(self._windowsize))
            twod_patches = patches.reshape(patches.shape[0], -1)
            normalized_patches = safe_unit_norm(twod_patches)
            dist = cdist(normalized_patches, self._templates).min(axis=0)
            dist[dist == 0] = 1e-2
            dist = 1. / dist
            out[i] = dist
        print out.shape
        yield ConstantRateTimeSeries(out, data.frequency, data.duration)
