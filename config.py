from model.pattern import Pattern
from model.pipeline import Pipeline
from data.pattern import InMemory
from data.learn import LearningController

# Data backends
# TODO: I don't like the redundancy here, i.e., the model class is a key in 
# the doctionary, *and* must be passed to the data controller
data = {
    Pattern  : InMemory(Pattern),
    Pipeline : LearningController(Pipeline)
}

# Audio config
samplerate = 44100
windowsize = 2048
stepsize   = 1024

# FrameModel
from analyze.feature import RawAudio, FFT, Loudness

class FrameModel(object):
    pass