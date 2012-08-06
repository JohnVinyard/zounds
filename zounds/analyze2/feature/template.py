import numpy as np
from zounds.nputil import safe_unit_norm as sun
from scipy.spatial.distance import cdist

from zounds.analyze2.extractor import SingleInput
from zounds.model.pipeline import Pipeline
from multiprocessing import Pool
from zounds.nputil import toeplitz2dc
from zounds.util import flatten2d
from zounds.learn.nnet.nnet import sigmoid

def toeplitz2d2(patch,size,silence_thresh):
    l = toeplitz2dc(patch.astype(np.float32),size)
    nz = l.sum(1) > silence_thresh
    # take the unit norm of the patches that are above the loudness threshold
    return sun(l[np.nonzero(nz)[0]])


def toeplitz2d(patch,size,silence_thresh):
    '''
    Construct a matrix that makes convolution possible via a matrix-vector
    operation, similar to constructing a toeplitz matrix for 1d signals
    '''
    # width of the patch
    pw = patch.shape[0]
    # total size of the patch
    patchsize = patch.size
    # width of the kernel
    kw = size[0]
    # height of the kernel
    kh = size[1]
    # size of the kernel
    ksize = np.product(size)
    
    # the patch width, without boundaries
    w = patch.shape[0] - kw
    # the patch height, without boundaries
    h = patch.shape[1] - kh
    # the total number of positions to visit
    totalsize = w * h
    # this will hold each section of the patch
    l = np.zeros((totalsize,ksize))
    c = 0

    # Make sure the entire patch has minimum of zero, since we're going
    # to be determining if sub-patches are louder than the silence threshold
    # by summing them.
    pzeroed = patch - patch.min()
    for s in range(patchsize):
        j = int(s / pw)
        i = s % pw
        if i < w and j < h:
            l[c] = pzeroed[i : i + kw, j : j + kh].ravel()
            c += 1

    # get the positions which contain "valid" sub-patches, i.e., the
    # sub-patches are louder than the silence threshold
    nz = l.sum(1) > silence_thresh
    # Give each sub-patch unit-norm. Return the unit-normed sub-patches 
    # and the "valid" indices
    return sun(l),np.nonzero(nz)[0]



def feature(args):
    patch,size,thresh,codebook,weights,activation,act_thresh = args
    
    out = np.ndarray((patch.shape[0],codebook.shape[0]))
    for i in xrange(patch.shape[0]):
        mat = toeplitz2d2(patch[i], size, thresh)
        if not mat.shape[0]:
            return np.zeros(out.shape)
        
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
        # know nothing about the Pipeline class
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
        
        # TODO: Make this an __init__ parameter
        self.feature_func = feature
    
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
        return [self.feature_func(arg) for arg in self._args]
        
    def _process_multi(self,data):
        pool = Pool(self._nbooks)
        results = pool.map(self.feature_func,self._args)
        pool.close()
        return results
    
    def _process(self):
        data = self.in_data
        self._update_args(data)
        out = self.__process(data)
        return np.concatenate(out,axis = 1)
