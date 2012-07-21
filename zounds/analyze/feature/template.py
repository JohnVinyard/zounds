import numpy as np
from zounds.nputil import safe_unit_norm as sun
from scipy.spatial.distance import cdist

from zounds.analyze.extractor import SingleInput
from zounds.model.pipeline import Pipeline
from multiprocessing import Pool

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
    mat,nz = toeplitz2d(patch, size, thresh)
    if not len(nz):
        return np.zeros(len(codebook))
    
    mat = mat[nz]
    if weights is not None:
        mat *= weights
    dist = cdist(mat,codebook)
    return activation(dist,act_thresh)


def _soft_activation(dist,act_thresh):
    '''
    Return 1 / distance, so that templates with the lowest distances
    have the highest values
    '''
    # variations on the triangle activation function recommended here:
    # http://robotics.stanford.edu/~ang/papers/nipsdlufl10-AnalysisSingleLayerUnsupervisedFeatureLearning.pdf
    dist[dist == 0] = 1e-3
    dist = (1. / dist)
    dist -= dist.mean(1)[:,np.newaxis]
    dist[dist < 0] = 0
    return dist.max(0)
    
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
        self._codebooks = [sun(cb) for cb in self._codebooks]
        # the output dimension will be the combined size of all the codebooks
        self._dim = np.sum([len(cb) for cb in self._codebooks])
        self._dtype = np.float32 if soft_activation else np.uint8
        self._sizes = sizes
        self._silence_thresh = silence_thresholds
        self.activation = \
            _soft_activation if soft_activation else _hard_activation
        self._hard_activation_thresh = hard_activation_thresh
        self.__process = \
            self._process_multi if multiprocess else self._process_single
        
        self._args = self._build_args(None)
        self._pool = None
        
    
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
        return [self._pool.map(feature,self._args)]
    
    def _process(self):
        if self._pool is None:
            self._pool = Pool(self._nbooks)
        
        data = np.array(self.in_data[:self.nframes]).reshape(self._inshape)
        self._update_args(data)
        return [np.concatenate(self.__process(data)).ravel()]
              
