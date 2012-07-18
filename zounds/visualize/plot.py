from matplotlib import pyplot as plt
import subprocess
import numpy as np

# TODO: How does this work on other systems, where matplotlib is working correctly?
def plot(arr,filename,figsize = (5,5), oned = False, twod = False, gray = False):
    if gray:
        plt.gray()
    else:
        plt.jet()
    arr = np.array(arr).squeeze()
    plt.figure(figsize = figsize)
    if oned or 1 == len(arr.shape):
        plt.plot(arr)
    elif twod or 2 == len(arr.shape):
        plt.matshow(np.rot90(arr))
    else:
        raise ValueError('arr must have 1, 2, or 3 dimensions')
    plt.show()
    plt.savefig(filename)
    plt.clf()

def plot_series(arr,filename,figsize = (5,5), oned = False, 
                twod = False, gray = False):
    '''
    Create several plot images, appending a unique number to each image file
    name. The size of dimension zero determines the number of plots to create.
    '''
    for i,a in enumerate(arr):
        plot(a,'%s_%i' % (filename,i),figsize=figsize,oned=oned,twod=twod,gray=gray)

def plot_vars(d,names,figsize=(5,5),oned = False, twod = False, gray = False):
    '''
    Useful for plotting variables defined in the current context, e.g., if two
    arrays named 'a' and 'b' are defined:
    
    plot_vars(globals(),['a','b'])
    '''
    for n in names:
        plot(d[n],n,figsize=figsize,oned=oned,twod=twod,gray=gray)
            

def video(arr,filename,audiofile = None,rate = 10):
    # ensure that the minimum value is zero
    rescaled = arr - arr.min()
    # ensure that the maximum value is 2**8
    rescaled = (rescaled * (255. / rescaled.max())).astype(np.uint8)
    vs = VideoSink(rescaled.shape[1:],filename = filename, 
                   audiofile = audiofile, rate = rate)
    for a in rescaled:
        vs.run(a)
    vs.close()
    

class VideoSink(object) :

    def __init__( self, size, filename='output', rate=10, byteorder='Y8', audiofile = None):
        self.size = size
        cmdstring  = ['mencoder',
                    '/dev/stdin',
                    '-demuxer', 'rawvideo',
                    '-rawvideo', ('w=%i:h=%i'%size[::-1]) + \
                    (':fps=%i:format=%s'%(rate,byteorder)),
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
