
# Audio config
class AudioConfig:
    samplerate = 44100
    windowsize = 2048
    stepsize   = 1024

# User Config
source = 'John'

# Extractor Chain
from analyze.feature import FFT, Loudness

# FrameModel
# TODO: FrameModel needs to have a metaclass that knows how to map extractors
# to properties, and map those properties to some database
from model.frame import Frames, Feature

class FrameModel(Frames):    
    fft      = Feature(FFT, store = True, needs = None)
    loudness = Feature(Loudness, store = True, needs = fft)
    

# Data backends
from model.pattern import Pattern
from model.pipeline import Pipeline
from data.pattern import InMemory
from data.learn import LearningController
from data.frame import FrameController
data = {
    Pattern    : InMemory,
    Pipeline   : LearningController,
    # TODO: The other controllers can know explicitly about the the classes
    # they're expected to return. Not so with the user-defined FrameModel
    # class. What are the implications of this?
    FrameModel : FrameController
}


from environment import Environment
# TODO: This has to work with multi-threaded and/or multi-process applications
Z = Environment(source,FrameModel,data,AudioConfig)

if __name__ == '__main__':
    
    print Pattern
    print Pattern.controller() 
    print Pipeline
    print Pipeline.controller()
