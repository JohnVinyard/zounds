# import zounds' logging configuration so it can be used in this application
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
from zounds.model.pipeline import Pipeline
from zounds.data.frame import ${ControllerClassName}
from zounds.data.search import PickledSearchController
from zounds.data.pipeline import PickledPipelineController

data = {
    ExhaustiveSearch    : PickledSearchController(),
    Pipeline            : PickledPipelineController()
}


from zounds.environment import Environment
dbfile = '${DbFile}'
Z = Environment(
                source,                  # name of this application
                FrameModel,              # our frame model
                ${ControllerClassName},  # FrameController class
                (FrameModel,dbfile),     # FrameController args
                data,                    # data-backend config
                audio = AudioConfig)     # audio configuration     

