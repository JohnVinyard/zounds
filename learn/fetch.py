from __future__ import division
from abc import ABCMeta,abstractmethod

import numpy as np

from environment import Environment
from util import pad

class Fetch(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        object.__init__(self)
    
    @abstractmethod
    def __call__(self, nexamples = None):
        '''
        Acquire a number of training examples
        '''
        pass

class NoOp(Fetch):
    
    def __init__(self):
        Fetch.__init__(self)
    
    def __call__(self, nexamples = None):
        return None

class PrecomputedFeature(Fetch):
    '''
    Fetches "patches" of precomputed features from the frames database.  Attempts
    to sample evenly from the database, drawing more samples from longer 
    patterns.
    '''
    def __init__(self,nframes,feature):
        Fetch.__init__(self)
        self.feature = feature
        self.nframes = nframes
        
    
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
        shape = (nexamples,axis1) if axis1 > 1 else (nexamples,)
        data = np.ndarray(shape,dtype = dtype)
        
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
                    print 'drawing %i samples from %s' % (n_to_draw,_id)
                    try:
                        s = np.random.randint(0,lf - self.nframes,n_to_draw)
                    except ValueError:
                        s = [0]
                    feature = pad(frames[self.feature],self.nframes)
                    for i in s:
                        data[nsamples] = \
                            feature[i : i + self.nframes]\
                            .reshape(axis1)
                        nsamples += 1
        except IndexError:
            # We tried to write to an index outside the bounds of data, which
            # means we're done collecting data.
            pass
                    
        # finally, since the patches are grouped by pattern,
        # shuffle them so they're completely randomized
        return data[np.random.permutation(len(data))]


            