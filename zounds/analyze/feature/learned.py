from __future__ import division
import numpy as np
from zounds.environment import Environment
from zounds.analyze.extractor import SingleInput
from zounds.model.pipeline import Pipeline
from itertools import product
from zounds.util import flatten2d
from zounds.nputil import norm_shape

# TODO: Write tests!
class Learned(SingleInput):
    '''
    A thin wrapper around a learned feature
    '''
    
    def __init__(self,
                  pipeline_id = None,
                  dim = None,
                  dtype = None,
                  needs = None, 
                  nframes = 1, 
                  step = 1, 
                  key = None):
        
        
        SingleInput.__init__(\
                        self, needs=needs,nframes=nframes,step=step,key=key)
        self.pipeline = Pipeline[pipeline_id]
        self._dim = dim
        self._dtype = dtype
        self.env = Environment.instance
    
    
    def dim(self,env):
        return self._dim
    
    @property
    def dtype(self):
        return self._dtype
    
    def _process(self):
        return self.pipeline(flatten2d(self.in_data))
    
# TODO: Generalize this to take any extractor, not just a learned one.
# I'm being held back by the fact that it's difficult to create extractors
# outside the context of an extractor chain.  This requires that I create one,
# and that it live and work in isolation.
class Tile(SingleInput):
    
    def __init__(self, needs = None, key = None, nframes = 1, step = 1,
                 pipeline_id = None, inshape = None, slicedim = None,
                 dtype = None, out_tile_shape = None,ravel = False):
        
        '''
        "Convolves" a pipeline with input by tiling it across the input, 
        activating it once for each tile, and concatentating the results.
        
        pipeline_id - The id of a stored Pipeline instance
        inshape - The shape that input data should be in before tiling begins
        slicedim -  A tuple specifying the size of tiles
        dtype - The output datatype
        out_tile_shape - Should be specified only if the Pipeline instance 
                        doesn't make its output dimension explicit
        ravel - if True, output is flattened, otherwise, it is returned as
                number of tiles x out_tile_shape
        '''
        SingleInput.__init__(self,needs = needs,nframes = nframes, step = step, key = key)
        self.pipeline = Pipeline[pipeline_id]
        self._inshape = norm_shape(inshape)
        self._slicedim = norm_shape(slicedim)
        self._ravel = ravel
        # ensure that inshape is evenly divisible by slicedim, element-wise
        if np.any(np.mod(inshape,slicedim)):
            raise ValueError('All dimensions of inshape must be evenly \
                            divisible by the respective slicedim dimensions')
        
        # the number of steps in each dimension
        self._nsteps = np.divide(inshape,slicedim)
        # the total number of tiles
        self._ntiles = np.product(self._nsteps)
        if None is not out_tile_shape:
            self._out_tile_shape = norm_shape(out_tile_shape)
        else:
            try:
                self._out_tile_shape = norm_shape(self.pipeline.dim)
            except NotImplemented:
                raise ValueError('You must either specify out_tile_shape, \
                                or use a Pipeline instance that implements the\
                                dim property')
        if ravel:
            # flatten the output
            self._dim = \
                np.product((self._ntiles,np.product(self._out_tile_shape)))
        else:
            # return an ntiles x out_tile_shape array
            self._dim = \
                (self._ntiles,) + tuple(np.array(self._out_tile_shape).ravel())
        
        self._dtype = dtype
        # generate all slices for each dimension
        self._slices = [[slice(j,j+self._slicedim[i]) \
                         for j in range(0,self._inshape[i],self._slicedim[i])] \
                         for i in range(len(self._inshape))]
        
        
        self._slice_prod = list(product(*self._slices))
        self._nslices = len(self._slice_prod)

    
    def dim(self,env):
        return self._dim

    @property
    def dtype(self):
        return self._dtype
    
    
    def _process(self):
        # reshape incoming data as necessary
        # Get each slice from each example and flatten the list into 2d
        # pass the entire list to the learning algorithm
        # reshape the output to be (nexamples,self._dim)
        
        data = self.in_data
        l = data.shape[0]
        data = np.reshape((l,) + self._inshape)
        all_slices = np.ndarray((l*self._nslices,) + self._slicedim)
        
        
        data = np.reshape(self.in_data[:self.nframes],self._inshape)
        # activate the pipeline on each slice of the input
        a = np.array([self.pipeline(data[coords].ravel()) \
                      for coords in product(*self._slices)]).astype(self._dtype)
        return a.ravel() if self._ravel else a
        
        
        