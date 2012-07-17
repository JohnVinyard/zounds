# User Config
source = '${Source}'

# Audio Config
class AudioConfig:
    samplerate = 44100
    windowsize = 2048
    stepsize = 1024
    window = None

# FrameModel
from model.frame import Frames, Feature


class FrameModel(Frames):
    raise Exception('Add some features here!')


# Data backends
from model.pattern import Pattern
from model.pipeline import Pipeline
from model.framesearch import MinHashSearch,LshSearch,ExhaustiveSearch

from data.pipeline import PickledPipelineController
from data.pattern import InMemory
from data.frame import PyTablesFrameController
from data.search import PickledSearchController

data = {
            
    Pattern             : InMemory(),
    Pipeline            : PickledPipelineController(),
    MinHashSearch       : PickledSearchController(),
    LshSearch           : PickledSearchController(),
    ExhaustiveSearch    : PickledSearchController()
}


from environment import Environment
dbfile = '${Directory}/datastore/frames.h5'
Z = Environment(
                source,                             # name of this application
                FrameModel,                         # our frame model
                PyTablesFrameController,            # FrameController class
                (FrameModel,dbfile),                # FrameController args
                data,                                # data-backend config
                audio = AudioConfig)                               

