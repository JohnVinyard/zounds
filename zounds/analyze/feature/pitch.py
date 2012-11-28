from __future__ import division

import numpy as np

from zounds.analyze.extractor import SingleInput

class YIN(SingleInput):
    '''
    Implementation of the YIN pitch detection algorithm, detailed here:
    http://recherche.ircam.fr/equipes/pcm/cheveign/pss/2002_JASA_YIN.pdf
    '''
    def __init__(self, needs = None, key = None, nframes = 1, step = 1):
        SingleInput.__init__(self, needs = needs, nframes = nframes,
                             step = step, key = key)
    @property
    def dtype(self):
        return np.float32
    
    def dim(self,env):
        return ()
    
    def _process(self):
        
        # data is audio samples over a window-size defined by the environment's
        # audio parameters
        data = self.in_data
        
        # perform auto-correlation
        # TODO: Should I just expect input from the AutoCorrelation feature?
        corr = np.correlate(data,data,mode = 'full')[data.shape[0]:]
        
        
        
        