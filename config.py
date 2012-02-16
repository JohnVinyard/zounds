
# Data backends
pattern_controller = None
frame_controller = None
learn_controller = None

# Audio config
samplerate = 44100
windowsize = 2048
stepsize = 1024

# FrameModel
from analyze.feature import RawAudio, FFT, Loudness

class FrameModel(object):
    pass