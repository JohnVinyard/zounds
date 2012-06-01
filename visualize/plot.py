from matplotlib import pyplot as plt
import subprocess
import numpy as np

def plot(arr,filename,figsize = (5,5)):
    arr = np.array(arr)
    plt.figure(figsize = figsize)
    if 1 == len(arr.shape):
        plt.plot(arr)
    elif 2 == len(arr.shape):
        plt.matshow(np.rot90(arr))
    else:
        raise ValueError('arr must have 1 or 2 dimensions')
    
    plt.show()
    plt.savefig(filename)
    plt.clf()
    

class VideoSink(object) :

    def __init__( self, size, filename="output", rate=10, byteorder="bgra", audiofile = None):
        self.size = size
        cmdstring  = ['mencoder',
                    '/dev/stdin',
                    '-demuxer', 'rawvideo',
                    '-rawvideo', 'w=%i:h=%i'%size[::-1]+":fps=%i:format=%s"%(rate,byteorder),
                    '-o', filename+'.avi',
                    '-ovc', 'lavc']
            
        if audiofile:
            cmdstring.extend(['-audiofile',audiofile,'-oac','pcm'])
        self.p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE, shell=False)

    def run(self, image) :
        assert image.shape == self.size
        self.p.stdin.write(image.tostring())

    def close(self) :
        self.p.stdin.close()
