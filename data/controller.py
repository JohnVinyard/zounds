

class Controller(object):
    '''
    Base class for all data controllers. These classes are responsible for
    persisting and retrieving objects defined in model.
    '''
    
    # TODO: I don't want to have to pass in a class here, but it's the only
    # way I can see, for now, to avoid a circular reference that goes something
    # like:
    #
    # config -> data - > model -> config
    #
    # I'm breaking data's dependence on model by instantiating a controller
    # with a model class in config
    def __init__(self,model_class):
        object.__init__(self)
        self.cls = model_class