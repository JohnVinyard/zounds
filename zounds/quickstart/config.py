from zounds.log import *

# User Config
source = '${Source}'

# Audio Config
class AudioConfig:
    samplerate = 44100
    windowsize = 2048
    stepsize = 1024
    window = None

# FrameModel
from zounds.model.frame import Frames, Feature
from zounds.analyze.feature.spectral import FFT,BarkBands

class FrameModel(Frames):
    fft = Feature(FFT, store = False)
    bark = Feature(BarkBands, needs = fft, nbands = 100, stop_freq_hz = 12000)


# Data backends
from zounds.model.framesearch import ExhaustiveSearch
from zounds.data.frame import PyTablesFrameController
from zounds.data.search import PickledSearchController

data = {
    ExhaustiveSearch    : PickledSearchController()
}


from zounds.environment import Environment
dbfile = 'datastore/frames.h5'
Z = Environment(
                source,                             # name of this application
                FrameModel,                         # our frame model
                PyTablesFrameController,            # FrameController class
                (FrameModel,dbfile),                # FrameController args
                data,                                # data-backend config
                audio = AudioConfig)                               

