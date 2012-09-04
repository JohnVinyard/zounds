from __future__ import division
import numpy as np
from zounds.environment import Environment
from zounds.analyze.extractor import SingleInput
from zounds.model.pipeline import Pipeline
from zounds.nputil import norm_shape,sliding_window,flatten2d

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
                (np.product((self._ntiles,np.product(self._out_tile_shape))),)
        else:
            # return an ntiles x out_tile_shape array
            self._dim = \
                (self._ntiles,) + tuple(np.array(self._out_tile_shape).ravel())
        
        self._dtype = dtype

    def dim(self,env):
        return self._dim

    @property
    def dtype(self):
        return self._dtype
        
    def _process(self):
        data = self.in_data
        # get the number of incoming samples
        l = data.shape[0]
        # reshape as specified
        data = np.reshape((l,) + self._inshape)
        # Make a list of every slice from every example
        examples = flatten2d(sliding_window(data,(1,) + self._slicedim))
        # activate the pipeline on each slice
        transformed = self.pipeline(examples)
        # reshape so that the first dimension once again represents examples
        return transformed.reshape((l,) + self._dim)
        
        
        