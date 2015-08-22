from flow import Node, NotEnoughData
import marshal
import types

class Op(object):
    
    def __init__(self, func, **kwargs):
        super(Op, self).__init__()
        self._kwargs = kwargs
        self._func = marshal.dumps(func.func_code)
        
    def __getattr__(self, key):
        if key == '_kwargs':
            raise AttributeError()
        
        try:
            return self._kwargs[key]    
        except KeyError: 
            raise AttributeError(key)
    
    def __call__(self, arg):
        code = marshal.loads(self._func)
        f = types.FunctionType(\
           code, 
           globals(), 
           'preprocess')
        return f(arg, **self._kwargs)

class PreprocessResult(object):
    
    def __init__(self, data, op):
        super(PreprocessResult, self).__init__()
        self.data = data
        self.op = op

class Preprocessor(Node):
    
    def __init__(self, needs = None):
        super(Preprocessor, self).__init__(needs = needs)
    
    def _extract_data(self, data):
        if isinstance(data, PreprocessResult):
            return data.data
        return data

class MeanStdNormalization(Preprocessor):        
    
    def __init__(self, needs = None):
        super(MeanStdNormalization, self).__init__(needs = needs)
    
    def _process(self, data):
        data = self._extract_data(data)
        mean = data.mean(axis = 0)
        std = data.std(axis = 0)
        
        def x(d, mean = None, std = None):
            return (d - mean) / std
        
        op = Op(x, mean = mean, std = std)
        data = op(data)
        yield PreprocessResult(data, op)

class UnitNorm(Preprocessor):
    
    def __init__(self, needs = None):
        super(UnitNorm, self).__init__(needs = needs)
    
    def _process(self, data):
        data = self._extract_data(data)
        def x(d):
            from zounds.nputil import safe_unit_norm
            return safe_unit_norm(d.reshape(d.shape[0], -1))
        
        op = Op(x)
        data = op(data)
        yield PreprocessResult(data, op)

class PreprocessingPipeline(Node):
    
    def __init__(self, needs = None):
        super(PreprocessingPipeline, self).__init__(needs = needs)
        self._pipeline = [None] * len(self._needs)
        
    def _enqueue(self, data, pusher):
        # BUG: the memory address of _needs and pusher won't be the same.
        # the former is a Feature instance, and the latter is an 
        # Extractor instance
        i = self._needs.index(pusher)
        if i < 0: raise Exception('pusher not found')
        self._pipeline[i] = data.op
    
    def _compute_op(self, pipeline):
        def x(d, pipeline = None):
            for p in pipeline:
                d = p(d)
            return d
        return Op(x, pipeline = pipeline)
    
    def _dequeue(self):
        if not self._finalized or not all(self._pipeline):
            raise NotEnoughData()
        
        return self._compute_op(pipeline = self._pipeline)
        
            