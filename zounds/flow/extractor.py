from itertools import izip
import contextlib
import inspect

class Node(object):

    def __init__(self,needs = None):
        super(Node,self).__init__()
        self._cache = None
        self._listeners = []
        self._finalized = False

        if needs is None:
            self._needs = []
        elif isinstance(needs,Node):
            self._needs = [needs]
        else:
            self._needs = needs

        for n in self._needs:
            n.add_listener(self)

    def __enter__(self):
        return self

    def __exit__(self,t,value,traceback):
        pass

    @property
    def is_root(self):
        return not self._needs

    def add_listener(self,listener):
        self._listeners.append(listener)

    def find_listener(self,predicate):
        for l in self._listeners:
            if predicate(l):
                return l
            else:
                return l.find_listener(predicate)
        return None

    def _enqueue(self,data,pusher):
        # TODO: decode data, and write it to the local queue
        self._cache = data

    def _dequeue(self):
        if self._cache is None:
            raise NotEnoughData()

        v = self._cache
        self._cache = None
        return v

    def _process(self,data):
        yield data

    def _finalize(self,pusher):
        self._finalized = True

    def _push(self,data):
        for l in self._listeners:
            # TODO: the listener may be remote, so process() cannot be called
            # directly
            [x for x in l.process(data,self)]

    def __finalize(self,pusher = None):
        self._finalize(pusher)
        for l in self._listeners:
            l.__finalize(self)

    def process(self,data = None,pusher = None):
        if data is not None:
            self._enqueue(data,pusher)

        try:
            data = self._process(self._dequeue())
            for d in data: yield self._push(d)
        except NotEnoughData:
            yield None

        if self.is_root:
            self.__finalize()
            self._push(None)
            yield None

class NotEnoughData(Exception):
    pass

class Graph(dict):

    def __init__(self,**kwargs):
        super(Graph,self).__init__(**kwargs)

    def roots(self):
        return dict((k,v) for k,v in self.iteritems() if v.is_root)

    def process(self,**kwargs):
        roots = self.roots()
        generators = [roots[k].process(v) for k,v in kwargs.iteritems()]
        with contextlib.nested(*self.values()) as _:
            [x for x in izip(*generators)]

    @staticmethod
    def from_locals():
        '''
        Build a graph from all Node instances in the caller's
        context
        '''
        l = inspect.stack()[1][0].f_locals
        return Graph(**dict((k,v) for k,v in l.iteritems() if isinstance(v,Node)))