
class BaseDimensionEncoder(object):
    def __init__(self, dim_type):
        super(BaseDimensionEncoder, self).__init__()
        self.dim_type = dim_type

    def matches(self, o):
        return isinstance(o, self.dim_type)

    def dict(self, o):
        raise NotImplementedError()

    def encode(self, o):
        return dict(type=self.dim_type.__name__, data=self.dict(o))


class BaseDimensionDecoder(object):
    def __init__(self, dim_type):
        super(BaseDimensionDecoder, self).__init__()
        self.dim_type = dim_type

    def matches(self, d):
        return d['type'] == self.dim_type.__name__

    def args(self, d):
        return tuple()

    def kwargs(self, d):
        return dict()

    def decode(self, d):
        data = d['data']
        return self.dim_type(*self.args(data), **self.kwargs(data))
