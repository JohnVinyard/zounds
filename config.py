from model.pattern import Pattern
from data.pattern import InMemory

# Data backends
data = {
    Pattern : InMemory(Pattern)
}

# Audio config
samplerate = 44100
windowsize = 2048
stepsize   = 1024

# FrameModel
from analyze.feature import RawAudio, FFT, Loudness

class FrameModel(object):
    pass