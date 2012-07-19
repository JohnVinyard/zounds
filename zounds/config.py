'''
Example configuration file
'''

# Audio config
class AudioConfig:
    samplerate = 44100
    windowsize = 2048
    stepsize   = 1024

# User Config
source = 'John'

# Extractor Chain
from analyze.feature.spectral import FFT, Loudness

# FrameModel
from model.frame import Frames, Feature

class FrameModel(Frames):    
    fft      = Feature(FFT,store = True)
    loudness = Feature(Loudness, store = True, needs = fft)
    

# Data backends
from model.pattern import Pattern
from data.pattern import InMemory
from data.frame import PyTablesFrameController
data = {
        
    Pattern    : InMemory()
}


from environment import Environment
Z = Environment(
                source,                             # name of this application
                FrameModel,                         # our frame model
                PyTablesFrameController,            # FrameController class
                (FrameModel,'datastore/frames.h5'), # FrameController args
                data,                               # data-backend config
                AudioConfig)                        # audio configuration


