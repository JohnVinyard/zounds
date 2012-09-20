from __future__ import division
from abc import ABCMeta,abstractmethod

import numpy as np

from zounds.environment import Environment
from zounds.nputil import pad
from zounds.util import tostring
from random import choice

class Fetch(object):
    '''
    Fetch is an abstract base class. Derived classes should implement the
    :code:`__call__` method which will fetch data from the data store in a manner \
    specified by the implementation.
    '''
    __metaclass__ = ABCMeta
    
    def __init__(self):
        object.__init__(self)
    
    @abstractmethod
    def __call__(self, nexamples = None):
        '''__call__
        
        Fetch n examples from the data store
        
        :param nexamples: The number of examples to fetch
        '''
        pass

class NoOp(Fetch):
    '''
    Occasionally, a "learned" feature might not need any training data, e.g.
    :py:class:`~zounds.learn.hash.minhash.MinHash`.  This class returns no data, \
     ever.
    '''
    
    def __init__(self):
        Fetch.__init__(self)
    
    def __call__(self, nexamples = None):
        return None

class PrecomputedFeature(Fetch):
    '''
    Fetches "patches" of pre-computed features from the database.  Attempts
    to sample evenly from the database, drawing more samples from longer 
    patterns.  Concretely, if this was our 
    :py:class:`~zounds.model.frame.Frames`-derived class::
    
        class FrameModel(Frames):
            fft = Feature(FFT)
            bark = Feature(BarkBands,needs = fft, nbands = 50)
    
    We could fetch 1,000 random examples of successive pairs of bark bands like
    so::
    
        >>> pf = PrecomputedFeature(2,FrameModel.bark)
        >>> samples = pf(1000)
        >>> samples.shape
        (1000,100)
    
    Note that the successive frames were flattened into vectors of dimension 100.
    '''
    def __init__(self,nframes,feature, reduction = None,filter = None):
        '''__init__
        
        :param nframes: The number of successive frames of feature to treat as \
        a single example
        
        :param feature: A *stored* feature instance, or its key
        
        :param reduction: An aggregate function (e.g. :code:`max()` or :code:`sum()`) \
        to apply example-wise to samples before returning them
        
        :param filter: A callable which takes the fetched samples, and returns \
        the indices that should be kept.  When filter is not :code:`None`, this \
        class is not guaranteed to return the number of samples requested.
        '''
        Fetch.__init__(self)
        self.feature = feature
        self.nframes = nframes
        self._reduction = reduction
        self._filter = filter

    @property
    def _feature_repr(self):
        try:
            return self.feature.key
        except AttributeError:
            return self.feature
            
    def __repr__(self):
        return tostring(self,short = False,feature = self._feature_repr,
                        nframes = self.nframes,reduction = self._reduction, 
                        filter = self._filter)
    
    def __str__(self):
        return tostring(self,feature = self._feature_repr,nframes = self.nframes)
        
    # TODO: I don't think this method works at all. Write some tests
    # and find out
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
                    if self._reduction:
                        buffer = self._reduction(buffer,axis = 0)
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
                    f = feature[i : i + self.nframes]
                    if self._reduction:
                        f = self._reduction(f,axis = 0)
                    yield nsamples, f.reshape(axis1)
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
        if not dim:
            axis1 = self.nframes
        elif self._reduction:
            axis1 = dim[0]
        else:
            axis1 = self.nframes * dim[0]
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
        if self._filter is not None:
            data = data[self._filter(data)]
        
        return data[np.random.permutation(len(data))]
        

class _Patch(object):
    
    __metaclass__ = ABCMeta
    
    def __init__(self):
        object.__init__(self)
    
    @abstractmethod
    def _slice(self):
        pass
    
    @abstractmethod
    def __call__(self,arr):
        pass

class Patch(_Patch):
    '''
    Specifies how to take *random* slices from a larger input, in one dimension.
    
    Concretely, let's say we'd like to take random slices of size 2 from an array
    of size 10.  Here's how :code:`Patch` would be used to achieve that.
    
    >>> p = Patch(0,10,2,step = 1)
    >>> a = np.arange(10)
    >>> p(a)
    array([0, 1])
    >>> p(a)
    array([4, 5])
    >>> p(a)
    array([2, 3])
    >>> p(a)
    array([4, 5])
    >>> p(a)
    array([1, 2])
    >>> p(a)
    array([2, 3])
    '''
    
    def __init__(self,start,stop,size,step = None):
        '''__init__
        
        :param start: The first position in the input where slices may begin
        
        :param stop: The position in the input that no slices may include
        
        :param size: The size of the slice
        
        :param step: The interval at which slices may begin.  If :code:`None`, \
        the step size defaults to the same value as the :code:`size` parameter.
        '''
        _Patch.__init__(self)
        self.fullsize = stop - start
        self.start = start
        self.step = step if step else size
        self.size = size
        self.stop = stop
        self._range = range(self.start,self.stop-(size-1),self.step)
    
    def _slice(self):
        start = choice(self._range)
        stop = start + self.size
        return slice(start,stop)
    
    def __call__(self,arr):
        return arr[self._slice()]

class NDPatch(_Patch):
    '''
    Specifies how to take *random* slices from a larger input, in d-dimensions.
    
    Concretely, let's say we'd like to tak random slices of size (2,2) from an \
    array of size (10,10).  Here's how :code:`NDPatch` would be used to achieve \
    that::
        
        >>> p = NDPatch(Patch(0,10,2,step = 1), Patch(0,10,2,step = 1))
        >>> a = np.arange(100).reshape((10,10)) # create a 10x10 array
        >>> p(a)
        array([[20, 21],
               [30, 31]])
        >>> p(a)
        array([[44, 45],
               [54, 55]])
        >>> p(a)
        array([[56, 57],
               [66, 67]])
        >>> p(a)
        array([[24, 25],
               [34, 35]])
    '''
    
    def __init__(self,*patches):
        '''__init__
        
        :param patches: A :py:class:`Patch` instance for each dimension
        '''
        _Patch.__init__(self)
        self.patches = patches
    
    def __iter__(self):
        return self.patches.__iter__()
    
    def _slice(self):
        return [p._slice() for p in self]
    
    def __call__(self,arr):
        return arr[self._slice()]
        
        

# TODO: Write tests
class PrecomputedPatch(PrecomputedFeature):
    '''
    Cut arbitrary patches from samples drawn from the database.  Concretely, one
    might want to draw constant-sized patches randomly from spectrograms, in both
    time and frequency.  Assuming this as the application's
    :py:class:`~zounds.model.frame.Frames`-derived class...::
    
        class FrameModel(Frames):
            fft = Feature(FFT)
            bark = Feature(BarkBands, needs = fft, nbands = 100)
    
    ...consider the case where we'd like to draw 10x10 patches which can occur
    anywhere, in both time and frequency::
        
        >>> time_patch = Patch(0,10,10)
        >>> freq_patch = Patch(0,100,10,step = 1)
        >>> pp = PrecomputedPatch(10,FrameModel.bark,(10,100),NDPatch(time_patch,freq_patch))
        >>> samples = pp(100)
        >>> samples.shape
        (100,10,10)
    '''
    def __init__(self,nframes,feature,fullsize,patch,filter = None):
        '''__init__
        
        :param nframes: The number of frames of :code:`feature` to fetch
        
        :param feature: A :py:class:`~zounds.model.frame.Feature`-derived \ 
        instance, or its key
        
        :param fullsize: An integer or tuple of integers representing the size \
        of the patch before slicing. It should be nframes x :code:`feature`'s \
        dimension
        
        :param patch: Can be one of the following:
        
            * A slice, if :code:`fullsize` is one-dimensional and the patches will be drawn from a static location
            * A list of slices, one for each dimension, if :code:`fullsize` is multi-dimensional and the patches will be drawn from a static location
            * A :py:class:`Patch` instance, if :code:`fullsize` is one-dimensional and the patches will be drawn from a random location
            * A :py:class:`NDPatch` instance, if :code:`fullsize` is multi-dimensional and the patches will be drawn from a random location

        :param filter: A callable which takes the fetched samples, and returns \
        the indices that should be kept.  When filter is not :code:`None`, this \
        class is not guaranteed to return the number of samples requested.
        '''
        PrecomputedFeature.__init__(self,nframes,feature)
        # TODO: Check that number of dimensions of fullsize and patch agree,
        # and that all slices are valid for fullsize
        
        # a tuple representing the desired shape of the data before "cutting"
        self.fullsize = fullsize
        
        self._patch = patch if isinstance(patch,list) else patch
        self._static = all(map(lambda i : isinstance(i,slice),self._patch))
        self._filter = filter
    
    # TODO: __str__ and __repr__ methods that show fullsize and patch size
    
    @property
    def patch(self):
        if self._static:
            return self._patch
        
        return self._patch._slice()
    
    def filter_data(self,data):
        
        if None is self._filter:
            # No filter is defined. Return the data as-is
            return data
        
        if callable(self._filter):
            # filter is a callable that returns indices from data to keep
            return data[self._filter(data)]
        
        # Assume that self._filter is a number
        axes = -np.arange(1,data.ndim)
        indices = np.nonzero(np.apply_over_axes(np.sum, data, axes) > self._filter)[0]
        return data[indices]
    
    
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
        print '-----------------------------------------------------------'
        data = self.filter_data(data)
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
        

            