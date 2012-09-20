import numpy as np
from zounds.analyze.extractor import Extractor
from zounds.nputil import flatten2d

class Composite(Extractor):
    '''
    Combine inputs from multiple extractors into a single feature. For example::
    
        class FrameModel(Frames):
            fft = Feature(FFT, store = False)
            bark = Feature(BarkBands, needs = fft, nbands = 100, stop_freq_hz = 12000, store = False)
            # scalar valued
            loudness = Feature(Loudness,needs = bark, store = False)
            # scalar valued
            centroid = Feature(SpectralCentroid, needs = bark, store = False)
            # scalar valued
            flatness = Feature(SpectralFlatness, needs = bark, store = False)
            # vector valued, dimension 13
            bfcc = Feature(BFCC, needs = bark, store = False, ncoeffs = 13)
            # Combine loudness and centroid scalar values into a vector of
            # dimension 2
            c1 = Feature(Composite, needs = [loudness,centroid])
            # Combine the scalar flatness value and the vector bfcc value
            # into a vector of dimension 14
            c2 = Feature(Composite, needs = [flatness,bfcc])
            # Combine the vector-valued c1 and c2 features into a vector of
            # dimension 16
            c3 = Feature(Composite, needs = [c1,c2])
    
    
    Note that multiple frames of features can be captured and collapsed into
    a Composite feature::
    
        class FrameModel(Frames):
            fft = Feature(FFT, store = False)
            loudness = Feature(Loudness, needs = fft)
            centroid = Feature(SpectralCentroid, needs = fft)
            # Collapse 5 frames of the scalar-valued loudness and centroid features
            # into vectors of dimension 10
            vec = Feature(Composite, needs = [loudness,centroid], nframes = 5, step = 5)
    '''
    
    def __init__(self,needs = None, step = 1, nframes = 1,key = None):
        '''__init__
        
        :param needs: An iterable of :py:class:`~zounds.model.frame.Feature` \
        instances
        
        :param nframes: The number of frames from the source feature needed to \
        perform a computation
        
        :param step: The number of frames of the input features described by a \
        single frame of this feature
        '''
        Extractor.__init__(\
                self,step = step, nframes = nframes, needs = needs, key = key)
    
    
    def dim(self,env):
        dims = [int(np.product(s.dim(env)) * self.nframes) for s in self.sources]
        return np.sum(dims)
    
    @property
    def dtype(self):
        return np.float32
    
    def _process(self):
        sources = []
        for source in self.sources:
            data = self.input[source]
            l = data.shape[0]
            dim = len(data.shape)
            data = data.reshape((l,1)) if 1 == dim else flatten2d(data)
            sources.append(data)
        return np.concatenate(sources,axis = 1)
    
    