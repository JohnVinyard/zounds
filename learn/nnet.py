import numpy as np
from matplotlib import pyplot as plt
from bitarray import bitarray
from random import choice
import struct

# BEGIN KLUDGE:
# I'm not sure this is the right place for this stuff

powers_of_2 = [i**2 for i in range(1024)]

def flipbit(x,bits=32):
    '''
    Flip one bit in x, at random, and return it
    '''
    return x ^ choice(powers_of_2[:bits])

fmts = {8 : 'B', 16 : 'H', 32 : 'I', 64 : 'L'}
def gethash(act):
    '''
    Take the binary activations of a hidden
    layer and encode them as a number
    '''
    try:
        fmt = fmts[len(act)]
    except KeyError:
        raise ValueError('act\'s length must correspond to the length\
in bits of a char,short,int, or long')

    b = bitarray()
    b.extend(act)
    return struct.unpack(fmt,b.tobytes())[0]

# END KLUDGE

def sigmoid(a):
    return 1. / (1 + np.exp(-a))

def stochastic_binary(a):
    return a > np.random.random_sample(a.shape)

def normalize_examplewise(samples):
    # normalize example-wise
    samples[samples == 0] = .0000001
    samples = samples.T
    samples *= .99 / samples.max(0)
    samples = samples.T
    return samples

def binarize(samples):
    bin = np.zeros(samples.shape)
    for i,s in enumerate(samples):
        current = s.copy()
        # find the top 10 freqs
        top = np.argsort(s)[-20:]      
        # inhibit the neighbors of those freqs
        inhibit = .5
        for t in top:
            if t > 0:
                current[t - 1] -= current[t] * inhibit
            if t < len(current) - 1:
                current [t + 1] -= current[t] * inhibit

        top = np.argsort(current)[-10:]      
        bin[i][top] = 1
        toolow = s < 2
        bin[i][toolow] = 0
    return bin

def binarize2(samples):
    bin = samples.copy()
    down = np.roll(bin,-1,axis=1)
    down[:,-1:] = 0
    up = np.roll(bin,1,axis=1)
    up[:,:1] = 0

    inhibit = .3
    bin -= down * inhibit
    bin -= up * inhibit

    thresh = 3
    bin[bin <= 3] = 0
    bin[bin > 3] = 1
    return bin

class nnet(object):

    def __init__(self):
        pass

    def activate(self,inp):
        raise NotImplemented('nnet is meant to be an abstract base class')

    def fromfeatures(self,features):
        '''
        Using the hidden or latent variables, return the reconstructed output
        '''
        raise NotImplemented('nnet is meant to be an abstract base class')

    def _show_1d(self,f):
        plt.plot(f)

    def _show_2d(self,f,shape):
        rc = np.sqrt(len(f))
        f -= f.min()
        f[f == 0] = .000001
        #plt.imshow(np.log(f.reshape(shape)).T)
        plt.imshow(f.reshape(shape).T)

    def show_filters(self,nfilters,twod=False,filename=None):
        # Always show a square set of filters
        rc = int(np.sqrt(nfilters))
        n = rc**2
        filters = np.random.permutation(self._weights.T)[:n]
        #plt.gray()
        plt.figure()
        for i in range(n):
            plt.subplot(rc,rc,i+1)
            f = filters[i]
            self._show_2d(f,twod) if twod else self._show_1d(f)
            plt.xticks(())
            plt.yticks(())
        if filename:
            plt.savefig(filename)
        plt.show()
        plt.clf()
        
    
    def do_recon(self,inp,twod=False,filename=None):
        out = self.activate(inp)
        n = out.shape[0]
        #plt.gray()
        plt.figure()
        
        # If we're doing 1d plots, it's more helpful to
        # view them overlapping, in the same plot
        ncols = 2 if twod else 1
        for i,o in enumerate(out):
            
            plt.subplot(n,ncols,i*ncols + 1)
            self._show_2d(inp[i],twod) if twod else self._show_1d(inp[i])
            plt.xticks(())
            plt.yticks(())

            if twod:
                # create a seperate plot for the reconstruction
                plt.subplot(n,ncols,i*ncols + 2)
            self._show_2d(o,twod) if twod else self._show_1d(o)
            if twod:
                # remove the ticks from the reconstruction plot
                plt.xticks(())
                plt.yticks(())

        if filename:
            plt.savefig(filename)
        plt.show()
        plt.clf()
            

