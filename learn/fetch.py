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
        
    # TODO: Write tests
    def _prob_one_fetch(self,controller,framemodel,dim,axis1,nexamples,prob):
        '''
        Fetch method to use when fetching every possible sample from the database.
        For nframes = 1, this just means every frame. For nframes > 1, it means
        overlapping windows if size nframes and a step size of one.
        '''
        c = controller
        bufferdim = 1 if not dim else dim[0]
        
        i = 0
        _ids = list(c.list_ids())
        for _id in _ids:
            buf = np.zeros((self.nframes,bufferdim))
            bi = 0
            for feature in c.iter_feature(_id,self.feature):
                buf[bi] = feature
                bi += 1
                if bi == self.nframes:
                    bi -=1
                    yield i,buffer.reshape(axis1)
                    buf = np.roll(buf,-1,0)
                    buf[-1] = 0
                    i += 1
    
    def _prob_lt_one_fetch(self,controller,framemodel,dim,axis1,nexamples,prob):
        '''
        Fetch method to use when fetching fewer examples than exist in the database.
        This draws a number of samples from each sound/pattern based on the 
        ratio of its length to the length of the entire database.
        '''
        FM = framemodel
        nsamples = 0
        # If our probability calculation is correct, we should only enter the
        # outer while loop once. 
        
        while nsamples < nexamples:
            ids = list(controller.list_ids())
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
                    yield nsamples,\
                            feature[i : i + self.nframes].reshape(axis1)
                    nsamples += 1
    
    def _get_env(self):
        '''
        Get contextual information
        '''
        env = Environment.instance
        FM = env.framemodel
        c = FM.controller()
        l = len(c)
        return FM,c,l
    
    def _get_shape(self,c,nexamples):
        '''
        using the feature, figure out what the shape and size of the data
        will be
        '''
        dim = c.get_dim(self.feature) 
        dtype = c.get_dtype(self.feature)
        # BUG: This assumes features will always be either one or two dimensions
        axis1 = self.nframes if not dim else self.nframes * dim[0]
        shape = (nexamples,axis1) if axis1 > 1 else (nexamples,)
        return dim,dtype,axis1,shape
        
    def __call__(self,nexamples = None):
        
        FM,c,l = self._get_env()
        
        if not nexamples:
            # sample every frame from the database
            nexamples = l 
            
        dim,dtype,axis1,shape = self._get_shape(c, nexamples)
        data = np.zeros(shape,dtype = dtype)
        
        # a rough estimate of the the probability that any possible frame will
        # be sampled, ignoring pattern boundaries and the tail end of the frames
        # database
        prob = nexamples / l
        
        if prob > 1:
            raise ValueError('More examples than frames in the frames database')
        
        fetch = self._prob_one_fetch if 1 == prob else self._prob_lt_one_fetch
        for i,d in fetch(c, FM, dim, axis1, nexamples, prob):
            try:
                data[i] = d
            except IndexError:
                # We tried to write to an index outside the bounds of data, which
                # means we're done collecting data.
                break
        return data[np.random.permutation(len(data))]
        

# TODO: Write tests
class PrecomputedPatch(PrecomputedFeature):
    '''
    Cut arbitrary patches from samples drawn from the database
    '''
    def __init__(self,nframes,feature,fullsize,patch):
        PrecomputedFeature.__init__(self,nframes,feature)
        
        # TODO: Check that number of dimensions of fullsize and patch agree,
        # and that all slices are valid for fullsize
        
        # a tuple representing the desired shape of the data before "cutting"
        self.fullsize = fullsize
        # a slice, or list of slices representing the "patch" to cut from the
        # full size feature
        if callable(patch):
            self._patch = patch
        elif isinstance(patch,slice):
            self._patch = [patch]
        elif isinstance(patch,list):
            self._patch = patch
        else:
            raise ValueError('patch must be a callable, slice, or a list of slices')
    
    @property
    def patch(self):
        if callable(self._patch):
            return self._patch()
        
        return self._patch
    
    def __call__(self, nexamples = None):
        FM,c,l = self._get_env()
        
        if not nexamples:
            # sample every frame from the database
            nexamples = l 
        
        
        dim,dtype,axis1,shape = self._get_shape(c, nexamples)
        datashape = [nexamples] + [s.stop - s.start for s in self.patch]
        data = np.zeros(datashape,dtype = dtype)
        
        # a rough estimate of the the probability that any possible frame will
        # be sampled, ignoring pattern boundaries and the tail end of the frames
        # database
        prob = nexamples / l
        
        if prob > 1:
            raise ValueError('More examples than frames in the frames database')
        
        fetch = self._prob_one_fetch if 1 == prob else self._prob_lt_one_fetch
        for i,d in fetch(c, FM, dim, axis1, nexamples, prob):
            try:
                data[i] = d.reshape(self.fullsize)[self.patch]
            except IndexError:
                # We tried to write to an index outside the bounds of data, which
                # means we're done collecting data.
                break
        return data[np.random.permutation(len(data))]

class StaticPrecomputedFeature(Fetch):
    '''
    Fetch samples at random once, using the PrecomputedFeature class, and 
    always return the same samples when fetch() is called. This class is
    only useful for comparisons of different learning algorithms, or different
    implementations of the same learning algorithm.
    '''
    
    def __init__(self,nframes,feature,nsamples):
        pcf = PrecomputedFeature(nframes,feature)
        self.samples = pcf(nsamples)
    
    def __call__(self, nexamples = None):
        '''
        Note that nexamples is ignored
        '''
        return self.samples
        

            