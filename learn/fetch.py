from __future__ import division
from abc import ABCMeta,abstractmethod

import numpy as np

from environment import Environment

class Fetch(object):
    
    def __init__(self):
        object.__init__(self)
    
    @abstractmethod
    def __call__(self):
        pass

class PrecomputedFeature(Fetch):
    
    def __init__(self,nframes,feature):
        Fetch.__init__(self)
        if self.feature.extractor().stepsize > 1:
            raise ValueError(\
                'Cannot fetch features with a step size greater than 1 at\
                 present')
        self.nframes = nframes
        self.feature = feature
    
        
        
    def __call__(self,nexamples = None):
        
        env = Environment.instance
        FM = env.framemodel
        c = FM.controller()
        l = len(c)
        
        if not nexamples:
            # sample every frame from the database
            nexamples = l 
            
        # using the feature, figure out what the shape and size of the data
        # will be
        dim = c.get_dim(self.feature) 
        dtype = c.get_dtype(self.feature)
        # BUG: This assumes features will always be either one or two dimensions
        axis1 = self.nframes if not dim else self.nframes * dim[0]
        data = np.ndarray((\
                nexamples,
                axis1),
                dtype = dtype)
         
        
        # a rough estimate of the the probability that any possible frame will
        # be sampled, ignoring pattern boundaries and the tail end of the frames
        # database
        prob = nexamples / l
        
        if 1 == prob:
            # TODO: return every frame of this feature shuffled
            raise NotImplemented()
        
        if prob > 1:
            raise ValueError('More examples than frames in the frames database')
        
        nsamples = 0
        # If our probability calculation is correct, we should only enter the
        # outer while loop once. 
        try:
            while nsamples < nexamples:
                ids = list(c.list_ids())
                while ids:
                    # choose an id at random and remove it from the list
                    index = np.random.randint(len(ids))
                    _id = ids.pop(index)
                    # KLUDGE: What if the pattern is too large to fit into memory?
                    frames = FM[_id]
                    # using our probability calculation, draw n samples from the
                    # pattern
                    lf = len(frames)
                    n_to_draw = int(lf * prob)
                    # always draw at least one sample
                    n_to_draw = n_to_draw if n_to_draw else 1
                    # BUG: What if the pattern isn't as long as nframes?
                    s = np.random.randint(0,lf - self.nframes,n_to_draw)
                    feature = frames[self.feature]
                    for i in xrange(n_to_draw):
                        data[nsamples + i] = \
                            feature[s : s + self.nframes]\
                            .reshape(axis1)
                        nsamples += 1
        except IndexError:
            # We tried to write to an index outside the bounds of data, which
            # means we're done collecting data.
            pass
                    
        
        return data
            