import numpy as np
from matplotlib import pyplot as plt

from nnet import NeuralNetwork,sigmoid,stochastic_binary as sb
from learn.learn import Learn


class Rbm(NeuralNetwork,Learn):
    '''
    classic, binary-binary rbm
    Ripped off from http://www.cs.toronto.edu/~hinton/code/rbm.m
    Justifications for different parameters can be found here:
    http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''
    def __init__(self,
                 indim,
                 hdim,
                 learning_rate=0.1,
                 weight_decay=.002,
                 initial_momentum=.5,
                 final_momentum=.9,
                 sparsity_target=.01,
                 sparsity_decay=.9,
                 sparsity_cost=.001):

        NeuralNetwork.__init__(self)
        Learn.__init__(self)
        
        # IO
        self._indim = indim
        self._hdim = hdim

        # sparsity
        self._sparsity_target = sparsity_target
        self._sparsity_decay = sparsity_decay
        self._sparsity_cost = sparsity_cost

        # initialize biases
        self._vbias = np.zeros(self._indim)
        self._hbias = np.zeros(self._hdim)
        if self._sparsity_target:
            self._hbias[:] = \
                np.log(self._sparsity_target / (1 - self._sparsity_target))

        # initialize the weights to a normal
        # distribution with 0 mean and 0.01 std
        self._weights = np.random.normal(0.0,
                                         .01,
                                         (self._indim,self._hdim))

        # learning rate, weight decay, momentum
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._initial_momentum = initial_momentum
        self._final_momentum = final_momentum
        
    @property
    def indim(self):
        return self._indim
    
    @property
    def hdim(self):
        return self._hdim

    @property
    def dim(self):
        return self.hdim

    def _up(self,v):
        '''
        Given a visible sample, return the pre-sigmoid and
        sigmoid activations of the hidden units
        '''
        pre_sigmoid = np.dot(v,self._weights) + self._hbias
        return pre_sigmoid,sigmoid(pre_sigmoid)

    def _h_from_v(self,v):
        '''
        Infer the state of the stochastic binary
        hidden units
        '''
        ps,s = self._up(v)
        return ps,s,sb(s)
    
    def _down(self,h):
        '''
        Given a hidden state, infer the activation of
        the visible units.
        '''
        pre_sigmoid = np.dot(h,self._weights.T) + self._vbias
        sig = sigmoid(pre_sigmoid)
        return pre_sigmoid,sig,sb(sig)
        

    def _v_from_h(self,h):
        '''
        At the moment, I'm following the pattern of the 
        rbm in the theano tutorial, but this is really just
        a superfluous wrapper
        '''
        ps,s,stoch = self._down(h)
        return ps,s,stoch

    def _gibbs_hvh(self,h):
        '''
        One step of gibbs sampling, starting from the
        hidden layer
        '''
        vps,vs,vstoch = self._v_from_h(h)
        ps,s,stoch = self._h_from_v(vs)
        return vs,ps,s,stoch

    def _positive_phase(self,inp):
        '''
        positive phase sample (reality)
        get the pre-sigmoid, sigmoid, and stochastic
        binary activations of the hidden units, given
        the visible activations
        '''
        ps,s,stoch = self._h_from_v(inp)
        posprod = np.dot(inp.T,s)
        pos_h_act = s.sum(axis=0)
        pos_v_act = inp.sum(axis=0)
        return stoch, posprod, pos_h_act, pos_v_act

    def _negative_phase(self,stoch):
        '''
        negative phase sample (fantasy)
        '''
        vs,gps,gs,gstoch = self._gibbs_hvh(stoch)
        negprod = np.dot(vs.T,gs)
        neg_h_act = gs.sum(axis=0)
        neg_v_act = vs.sum(axis=0)
        return vs, negprod, neg_h_act, neg_v_act

    def _update(self,inp,momentum,epoch,batch):
        '''
        Compare the energy of the input with the energy
        of a fantasy produced by beginning with that input and
        doing one step of gibbs sampling

        Update the weights by multiplying the difference between
        the input and fantasy free energies by a learning rate
        '''

        # positive phase (reality)
        stoch, posprod, pos_h_act, pos_v_act = self._positive_phase(inp)

        # negative phase (fantasy)
        v, negprod, neg_h_act, neg_v_act = self._negative_phase(stoch)

        # calculate reconstruction error
        error = np.sum(np.abs(inp - v)**2)

        # number of samples in this batch
        n = float(inp.shape[0])
        m = momentum
        lr = self._learning_rate
        wd = self._weight_decay

        # sparsity
        if None != self._sparsity_target:
            current_sparsity = stoch.sum(0) / float(n)
            self._sparsity = (self._sparsity_decay * self._sparsity) + \
                ((1 - self._sparsity_decay) * current_sparsity)
            sparse_penalty = self._sparsity_cost * \
                (self._sparsity_target - self._sparsity)
            

        # update the weights
        self._wvelocity = (m * self._wvelocity) + \
            lr * (((posprod-negprod)/n) - (wd*self._weights))
        self._weights += self._wvelocity
        if None != self._sparsity_target:
            self._weights += sparse_penalty

        # update the visible biases
        self._vbvelocity = (m * self._vbvelocity) + \
            ((lr/n) * (pos_v_act - neg_v_act))
        self._vbias += self._vbvelocity

        # update the hidden biases
        self._hbvelocity = (m * self._hbvelocity) + \
            ((lr/n) * (pos_h_act - neg_h_act))
        self._hbias += self._hbvelocity
        if None != self._sparsity_target:
            self._hbias += sparse_penalty
        
        return error


    def train(self,samples,stopping_condition):
        '''
        '''
        batch_size = 100
        nbatches = len(samples) / batch_size
        samples = samples.reshape((nbatches,batch_size,samples.shape[1]))
        # initialize some variables we'll
        # use during training, but then throw away
        self._wvelocity = np.zeros(self._weights.shape)
        self._vbvelocity = np.zeros(self._indim)
        self._hbvelocity = np.zeros(self._hdim)
        self._sparsity = np.zeros(self._hdim)
        
        epoch = 0
        error = 99999
        nbatches = len(samples)
        while not stopping_condition(epoch,error):
            if epoch < 5:
                mom = self._initial_momentum
            else:
                mom = self._final_momentum
            batch = 0
            while batch < nbatches and not stopping_condition(epoch,error):
                error = self._update(samples[batch],mom,epoch,batch)
                batch += 1
                print 'Epoch %i, Batch %i, Error %1.4f' % (epoch,batch,error)
            epoch += 1
        
        
        # get rid of the "temp" training variables
        del self._wvelocity
        del self._vbvelocity
        del self._hbvelocity
        del self._sparsity

    def activate(self,inp):
        '''
        Activate the net using the visible sample.
        Threshold the probabilities that the hidden
        units are on.
        '''
        hps,hs,hstoch = self._h_from_v(inp)
        vps,vs,vstoch = self._v_from_h(hs)
        vs[vs > .5] = 1
        vs[vs <=.5] = 0
        return vs

    def fromfeatures(self,features):
        raise NotImplemented()
    
    # TODO: Is this the correct implementation for both Rbm and LinearRbm?
    def __call__(self,data):
        ps,s,stoch = self._h_from_v(data)
        s[s > .5] = 1
        s[s <= .5] = 0
        return s

class LinearRbm(Rbm):

    def __init__(self,indim,hdim,
                 sparsity_target=.01,
                 learning_rate = .001):
        Rbm.__init__(self,
                     indim,
                     hdim,
                     learning_rate = learning_rate,
                     sparsity_target=sparsity_target)


    def _gibbs_hvh(self,h):
        '''
        One step of gibbs sampling, starting from the
        hidden layer
        '''
        vps,vs,vstoch = self._v_from_h(h)
        # use the actual value of the visible units,
        # instead of the value passed through a sigmoid function
        ps,s,stoch = self._h_from_v(vps)
        return vps,ps,s,stoch

    def activate(self,inp):
        hps,hs,hstoch = self._h_from_v(inp)
        vps,vs,vstoch = self._v_from_h(hs)
        return vps
                     
                     
import cPickle
import argparse
if __name__ == '__main__':


    parser = argparse.ArgumentParser(\
        description = 'Module to train binary rbms')
    aa = parser.add_argument


    aa('--type',
       help='either rbm or linear_rbm')
    aa('--fetch',
       help='the module which will fetch training data')
    aa('--paramfile',
       help='the base name of files in which parameters will be saved')
    aa('--guess',
       help='file with pickled initial parameter guess.  \
 This might be the output of a previous training session.  \
If not specified, start from a random guess.',
       type=str,
       default=None)
    aa('--train',
       help='train either a new rbm, or the one represented\
by --guess',
       default=False,
       action='store_true')

    aa('--epochs',
       help='the total number of epochs to train',
       default=15,
       type=int)
    aa('--nsamples',
       help='the number of samples to train on. \
       Must be evenly divisible by 100.',
       type=int)
    aa('--visdim',
       help='the number of visible units',
       type=int)
    aa('--hiddim',
       help='the number of hidden units',
       type=int)
    aa('--sparsity',
       help='the sparsity target',
       default=.01,
       type=float)
    aa('--twod',
       help='filters and reconstructions should be represented as a matrix,\
       rather than a 1d graph',
       default=False,
       action='store_true')
    args = parser.parse_args()

    fetch = __import__(args.fetch).fetch
    
    if args.train:
        nsamples = args.nsamples
        samples = fetch(nsamples)
        samples = np.array(samples,order='C')
        visdim = args.visdim
        hiddim = args.hiddim

        # reshape into mini-batches
        batchsize = 100
        samples = samples.reshape((nsamples/batchsize,batchsize,visdim))
        if args.guess:
            with open(args.guess,'rb') as f:
                net = cPickle.load(f)        
        else:
            net = eval(args.type)(visdim,hiddim,sparsity_target=args.sparsity)

    
        try:
            net.train(samples,args.epochs)
        except KeyboardInterrupt:
            # learning probably stalled out, and the user
            # hit ctrl-c to go ahead and save the weights
            # already
            pass

        print 'saving weights'
        with open(args.paramfile,'wb') as f:
            cPickle.dump(net,f)

    
    
    raise Exception('Size and dimensionality of filters and reconstructions\
    needs to be specified by command line args!')

    with open(args.paramfile,'rb') as f:
        net = cPickle.load(f)

    if args.twod:
        net.show_filters(1,twod=(),filename='bark_rbm_filters.png')
    else:
        net.show_filters(25,filename='bark_rbm_filters.png')
        
    print 'doing reconstructions'
    testsamples = fetch(100)
    torecon = testsamples[:1]
    if args.twod:
        net.do_recon(torecon,twod=(),filename='bark_rbm_recon.png')
    else:
        net.do_recon(torecon,filename='bark_rbm_recon.png')

    
    plt.gray()
    ps,s,stoch = net._h_from_v(testsamples)
    s[s > .5] = 1
    s[s <= .5] = 0
    plt.matshow(s)
    plt.show()
    plt.savefig('bark_rbm_hidden_activations.png')
    plt.clf()

