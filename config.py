
# Audio config
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
# TODO: I don't like the redundancy here, i.e., the model class is a key in 
# the dictionary, *and* must be passed to the data controller's constructor
from model.pattern import Pattern
from model.pipeline import Pipeline
from data.pattern import InMemory
from data.learn import LearningController
data = {
    Pattern  : InMemory(Pattern),
    Pipeline : LearningController(Pipeline)
}



