import numpy as np
from matplotlib import pyplot as plt

# TODO: package up these activation functions somehow
def sigmoid(a):
    return 1. / (1 + np.exp(-a))

def stochastic_binary(a):
    return a > np.random.random_sample(a.shape)

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
            

