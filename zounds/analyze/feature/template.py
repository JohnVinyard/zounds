from multiprocessing import Pool

import numpy as np
from scipy.spatial.distance import cdist

from zounds.nputil import safe_unit_norm as sun,sliding_window
from zounds.analyze.extractor import SingleInput
from zounds.model.pipeline import Pipeline
from zounds.nputil import flatten2d


def get_subpatches(patch,size,silence_thresh):
    # get subpatches from patch, with a stride of one in each direction
    l = flatten2d(sliding_window(patch,size,(1,) * len(size)))
    # find out which patches are above the loudness threshold
    nz = l.sum(1) > silence_thresh
    # only return those patches that are above the loudness threshold, giving
    # them all unit norm
    return sun(l[np.nonzero(nz)[0]])


def feature(args):
    patch,size,thresh,codebook,weights,activation,act_thresh = args
    out = np.ndarray((patch.shape[0],codebook.shape[0]))
    # TODO: Would it make sense to chunk patches and do the distance calculation
    # all-at-once? 
    for i in xrange(patch.shape[0]):
        mat = get_subpatches(patch[i],size,thresh)
        if not mat.shape[0]:
            out[i] = np.zeros(codebook.shape[0])
            continue
        
        if weights is not None:
            mat *= weights
        
        dist = cdist(mat,codebook)
        out[i] = activation(dist,act_thresh)
    
    return out


def _soft_activation(dist,act_thresh):
    '''
    Return 1 / distance, so that templates with the lowest distances
    have the highest values
    '''
    m = dist.min(0)
    m[m == 0] = 1e-2
    return 1. / m

def _hard_activation(dist,act_thresh):
    '''
    Pass the soft activation through a hard thresholding function
    '''
    act = _soft_activation(dist,act_thresh)
    return act <= act_thresh




class TemplateMatch(SingleInput):
    '''
    Convolve each template from any number of codebooks with a spectrogram
    '''
    def __init__(self,needs = None, key = None, nframes = None, step = None,
                 inshape = None, codebooks = None, weights = None, sizes = None, 
                 silence_thresholds = None, soft_activation = False,
                 hard_activation_thresh = 3,multiprocess = True):
        '''
        inshape   - the shape input data should be in before processing. Usually
                    (nframes,nbands of spectrogram)
        codebooks - A tuple of 2d codebooks, each containing templates to be 
                    convolved with the input spectrogram
        weights   - Weights to multiply each subpatch by before kmeans coding.
                    This probably means that whitening was performed prior to
                    learning the means. Input to be coded needs to be transformed
                    similarly.
        sizes     - A template dimension for each codebook. Codebooks are "flat",
                    i.e., 2d. These are the dimensions codes from each book should
                    be shaped into prior to convolution.
        silence_thresholds - For each codebook, a threshold below which a patch
                             from the input spectrogram will be ignored in the
                             distance calculation.
        soft_activation - If True, features will be 1 / distance for each 
                          patch,template pair. Otherwise, features will be 
                          passed through a thresholding function.
        hard_activation_thresh - If soft_activation is False, features will be
                                 binarized using this as the threshold.
        multiprocess - If True, a process will be launched for each codebook.
                       Convolutions and distance calculations will happen in 
                       parallel
        '''
        
        SingleInput.__init__(self, needs = needs, key = key, 
                             nframes = nframes, step = step)
        
        if weights is None:
            weights = len(codebooks)*[None]
        if 1 != len(set([len(codebooks),len(sizes),len(silence_thresholds),len(weights)])):
            raise ValueError(\
            'codebooks, sizes, and silence_thresholds must all have the same' +
            'number of elements')
        
        self._weights = weights
        # KLUDGE: I should be passing in numpy arrays directly. This class should
        # know nothing about the Pipeline class. This is a problem because numpy
        # arrays aren't pickleable, and therefore can't be passed as arguments
        # to Feature.__init__
        self._codebooks = [Pipeline[cb].learn.codebook for cb in codebooks]
        self._inshape = inshape
        self._nbooks = len(codebooks)
        # the output dimension will be the combined size of all the codebooks
        self._dim = np.sum([len(cb) for cb in self._codebooks])
        self._codebooks = [sun(cb) for cb in self._codebooks]
        
        
        self._dtype = np.float32 if soft_activation else np.uint8
        self._sizes = sizes
        self._silence_thresh = silence_thresholds
        self.activation = \
            _soft_activation if soft_activation else _hard_activation
        self._hard_activation_thresh = hard_activation_thresh
        self.__process = \
            self._process_multi if multiprocess else self._process_single
        
        self._args = self._build_args(None)
    
    @property
    def dtype(self):
        return self._dtype
    
    def dim(self,env):
        return self._dim
    
    def _build_args(self,data):
        return [[data,
                 self._sizes[i],
                 self._silence_thresh[i],
                 self._codebooks[i],
                 self._weights[i],
                 self.activation,
                 self._hard_activation_thresh]\
                for i in xrange(self._nbooks)]
        
    def _update_args(self,data):
        for i,arg in enumerate(self._args):
            self._args[i][0] = data
    
    def _process_single(self,data):
        return [feature(arg) for arg in self._args]
        
    def _process_multi(self,data):
        pool = Pool(self._nbooks)
        results = pool.map(feature,self._args)
        pool.close()
        return results
    
    def _process(self):
        data = self.in_data
        if data.shape == self._inshape:
            data = data.reshape((1,) + self._inshape)
        else:
            data = data.reshape((data.shape[0],) + self._inshape)
        self._update_args(data)
        out = self.__process(data)
        return np.concatenate(out,axis = 1)
