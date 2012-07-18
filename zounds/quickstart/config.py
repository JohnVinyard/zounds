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
    bark = Feature(BarkBands, nbands = 100, stop_freq_hz = 12000)


# Data backends
from zounds.model.pattern import Pattern
from zounds.model.pipeline import Pipeline
from zounds.model.framesearch import MinHashSearch,LshSearch,ExhaustiveSearch

from zounds.data.pipeline import PickledPipelineController
from zounds.data.pattern import InMemory
from zounds.data.frame import PyTablesFrameController
from zounds.data.search import PickledSearchController

data = {
            
    Pattern             : InMemory(),
    Pipeline            : PickledPipelineController(),
    MinHashSearch       : PickledSearchController(),
    LshSearch           : PickledSearchController(),
    ExhaustiveSearch    : PickledSearchController()
}


from zounds.environment import Environment
dbfile = '${Directory}/datastore/frames.h5'
Z = Environment(
                source,                             # name of this application
                FrameModel,                         # our frame model
                PyTablesFrameController,            # FrameController class
                (FrameModel,dbfile),                # FrameController args
                data,                                # data-backend config
                audio = AudioConfig)                               

