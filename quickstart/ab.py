from config import *
import cPickle
from random import choice
import sys


def load_data(filename):
    try:
        with open(filename,'r') as f:
            return cPickle.load(f)
    except IOError:
        return []

def save_data(data,filename):
    with open(filename,'w') as f:
        cPickle.dump(data, f)

def main(data,nseconds = 3):
    _ids = list(FrameModel.list_ids())
    q_id = choice(_ids)
    _ids.remove(q_id)
    a_id = choice(_ids)
    _ids.remove(a_id)
    b_id = choice(_ids)
    _ids.remove(b_id)
    


if __name__ == '__main__':
    filename = sys.argv[1]
    data = load_data(filename)
    try:
        while True:
            main()
    except:
        save_data(data,filename)
        