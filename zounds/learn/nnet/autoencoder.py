import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fmin_l_bfgs_b as bfgs
import cPickle
from nnet import NeuralNetwork,sigmoid

from zounds.learn.learn import Learn


class Params:
    
    '''
    A wrapper around weights and biases
    for an autoencoder
    '''
    def __init__(self,indim,hdim,w1 = None,b1 = None,w2 = None,b2 = None):
        self._indim = indim
        self._hdim = hdim
        
        # initialize the weights randomly and the biases to 0
        r = np.sqrt(6.) / np.sqrt((indim * 2) + 1.)
        self.w1 = w1 if w1 is not None else \
            np.random.random_sample((indim,hdim)) * 2 * r - r
        self.b1 = b1 if b1 is not None else np.zeros(hdim)
        
        self.w2 = w2 if w2 is not None else \
            np.random.random_sample((hdim,indim)) * 2 * r - r
        self.b2 = b2 if b2 is not None else np.zeros(indim)


    def unroll(self):
        '''
        "Unroll" all parameters into a single parameter vector
        '''
        w1 = self.w1.reshape(self._indim * self._hdim)
        w2 = self.w2.reshape(self._hdim * self._indim)
        return np.concatenate([w1,
                               w2,
                               self.b1,
                               self.b2])
    @staticmethod
    def from_file(filename):
        with open(filename,'rb') as f:
            return cPickle.load(f)

    @staticmethod
    def roll(theta,indim,hdim):
        '''
        "Roll" all parameter vectors into a params instance
        '''
        wl = indim * hdim
        w1 = theta[:wl]
        w1 = w1.reshape((indim,hdim))
        bb =  wl + wl
        w2 = theta[wl: bb]
        w2 = w2.reshape((hdim,indim))
        b1 = theta[bb : bb + hdim] 
        b2 = theta[bb + hdim : bb + hdim + indim]
        return Params(indim,hdim,w1,b1,w2,b2)

    def __eq__(self,o):
        return (self.w1 == o.w1).all() and \
            (self.w2 == o.w2).all() and \
            (self.b1 == o.b1).all() and \
            (self.b2 == o.b2).all()

    @staticmethod
    def _test():
        p = Params(100,23)
        a = p.unroll()
        b = Params.roll(a,100,23)
        print p == b
        print all(a == b.unroll())

        


class Autoencoder(NeuralNetwork,Learn):
    
    '''
    A horribly inefficient implementation
    of an autoencoder, which is meant to 
    teach me, and not much else.   
    '''

    def __init__(self, params, corruption = 0.0, linear_decoder=False):
        
        NeuralNetwork.__init__(self)
        Learn.__init__(self)
        
        self._include_weight_decay = True
        self._include_sparsity = True

        # this is multiplied against the sparsity delta
        self._beta = 3
        self._sparsity = 0.01
        
        # this is the learning rate
        self._alpha = .01

        # this is the weight decay term
        self._lambda = 0.0001

        self._netp = params

        # percentage of the inputs to corrupt
        # (i.e., set to zero). If this number
        # is greater than zero, then this becomes
        # a denoising autoencoder
        self._corruption = corruption

        # If true, don't apply the sigmoid function to the output
        self._linear_decoder = linear_decoder
        assert not self._linear_decoder


    @property
    def _weights(self):
        return self._netp.w1

    @property
    def indim(self):
        return self._netp._indim

    @property
    def hdim(self):
        return self._netp._hdim
    
    @property
    def dim(self):
        return self.hdim


    ## ACTIVATION OF THE NETWORK #########################################

    def _pre_sigmoid(self,a,w):
        '''
        Compute the pre-sigmoid activation
        for a layer 
        '''
        z = np.zeros((a.shape[0],w.shape[1]))
        for i,te in enumerate(a):
            z[i] = (w.T * te).sum(1)
        return z
        #return np.dot(a,w)

    def _sigmoid(self,la):
        '''
        compute the sigmoid function for an array
        of arbitrary shape and size
        '''
        return sigmoid(la)

    def _corrupt(self,inp):
        '''
        Return a corrupted version of
        the input. If the corruption
        percentage is zero, then return
        the unaltered input
        '''
        if 0 == self._corrupt:
            # this isn't a denoising autoencoder
            return inp

        # multiply the input by a binomial distribution
        # where the likelihood of a zero is self._corruption.
        # Each sample won't be corrupted by exactly the corruption
        # percentage, but something pretty close
        return inp * np.random.binomial(1,1 - self._corruption,(inp.shape))
        

    def fromfeatures(self,features,params = None):
        if None is params:
            params = self._netp

        m = features.shape[0]
        # pre-sigmoid activation of the output layer
        z3 = self._pre_sigmoid(features,params.w2) + np.tile(params.b2,(m,1))
        # sigmoid activation of the hidden layer,
        # or linear values of weights, if this is a linear decoder
        return self._sigmoid(z3) if not self._linear_decoder else z3

        
    def _activate(self,inp,params=None):
        '''
        Activate the net on a batch of training examples.
        return the input, and states of all layers for all
        examples.
        '''

        if None is params:
            params = self._netp

        # corrupt the input (if corrupt is 0, the input
        # is returned unchanged
        inp = self._corrupt(inp)

        # number of training examples
        m = inp.shape[0]
        # pre-sigmoid activation at the hidden layer
        z2 = self._pre_sigmoid(inp,params.w1) + np.tile(params.b1,(m,1))
        # sigmoid activation of the hidden layer
        a = self._sigmoid(z2)
        # pre-sigmoid activation of the output layer
        z3 = self._pre_sigmoid(a,params.w2) + np.tile(params.b2,(m,1))
        # sigmoid activation of the hidden layer,
        # or linear values of weights, if this is a linear decoder
        o = self._sigmoid(z3) if not self._linear_decoder else z3

        return inp,a,z2,z3,o

    def activate(self,inp):
        inp,a,z2,z3,o = self._activate(inp)
        return o
    
    def __call__(self,inp):
        '''
        Return the sigmoid activation of the hidden layer
        '''
        inp,a,z2,z3,o = self._activate(inp)
        return a

    ## PARAMETER LEARNING ################################################

    def _rcost(self,inp,params,a=None,z2=None,z3=None,o=None):
        '''
        The real-valued cost of using the parameters
        on a (batch) input
        '''
        if None == o:
            # activate the network with params
            ae = Autoencoder(params)
            inp,a,z2,z3,o = ae._activate(inp,params=params)
        

        # find the norm of the difference between each
        # input/output pair
        diff = o - inp
        n = np.sqrt((diff**2).sum(1))
        
        # one half squared error between input and output
        se = (0.5 * (n ** 2)).sum()
        
        # cost is the average of the one half squared error
        c = ((1. / inp.shape[0]) * se)

        # sum of squared error + weight decay + sparse
        if self._include_weight_decay:
            c += self._weight_decay_cost(params=params)

        if self._include_sparsity:
            c += self._sparse_cost(a)

        return c

    def _fprime(self,a):
        '''
        first-derivative of the sigmoid function
        '''
        return a * (1. - a)

    def _sparse_ae_cost(self,inp,parms,check_grad = False):
        '''
        Get the batch error and gradients for many
        training examples at once and update
        weights and biases accordingly
        '''

        inp,a,z2,z3,o = self._activate(inp,params=parms)
        d3 = -(inp - o)
        if not self._linear_decoder:
            d3 *= self._fprime(o)

        bp = self._backprop(d3,params=parms)
        spgd = np.zeros(bp.shape)
        if self._include_sparsity:
            spgd = self._sparse_grad(a)
            bp += spgd
        d2 = bp * self._fprime(a)

        wg2 = self._weight_grad(d3,a,parms.w2)
        wg1 = self._weight_grad(d2,inp,parms.w1)

        bg2 = self._bias_grad(d3)
        bg1 = self._bias_grad(d2)
        c = self._rcost(inp,parms,a=a,z2=z2,z3=z3,o=o)
        
        if check_grad:

            # unroll the gradients into a flat vector
            rg = Params(self.indim,self.hdim,wg1,bg1,wg2,bg2).unroll()

            # perform a (very costly) numerical 
            # check of the gradients
            self._check_grad(parms,inp,rg)

        return c, wg2, wg1, bg2, bg1

    def _sparse_ae_cost_unrolled(self,inp,parms):
        c, wg2, wg1, bg2, bg1 = self._sparse_ae_cost(inp,parms)
        return Params(self.indim,self.hdim,wg1,bg1,wg2,bg2).unroll()

    def _weight_decay_cost(self,params=None):
        if None is params:
            params = self._netp

        w1 = params.w1**2
        w2 = params.w2**2
        return (self._lambda / 2.) * (w1.sum() + w2.sum())

    def _sparse_cost(self,a):
        '''
        compute the sparsity penalty for a batch
        of activations.
        '''
        p = self._sparsity
        p_hat = np.average(a,0)
        print 'sparsity : %1.4f' % np.average(p_hat)
        a = p * np.log(p /p_hat)
        b = (1 - p) * np.log((1 - p) / (1 - p_hat))
        return self._beta * (a + b).sum()


    def _sparse_grad(self,a):
        p_hat = np.average(a,0)
        p = self._sparsity
        return self._beta * (-(p / p_hat) + ((1 - p) / (1 - p_hat)))

        
    def _bias_grad(self,cost):
        return (1. / cost.shape[0]) * cost.sum(0)

    def _weight_grad_lmem(self,cost,a,w):
        '''
        A limited-memory version of the weight gradient
        computation which uses a for loop instead of a 
        memory-intensive vectorized implementation
        '''
        wg = np.zeros(w.shape)
        for i in range(len(cost)):
            wg += np.outer(a[i],cost[i])
        wg = (1. / len(cost)) * wg
        if self._include_weight_decay:
            wg += (self._lambda * w)
        return wg

    def _weight_grad(self,cost,a,w):
        # compute the outer product of the error and the activations
        # for each training example, and sum them together to
        # obtain the update for each weight
        try:
            wg = (cost[:,:,np.newaxis] * a[:,np.newaxis,:]).sum(0).T
            wg = ((1. / cost.shape[0]) * wg)
            if self._include_weight_decay:
                wg += (self._lambda * w)
            return wg
        except ValueError:
            print 'switching to limited memory weight grad'
            self._weight_grad = self._weight_grad_lmem
            return self._weight_grad_lmem(cost,a,w)

    def _backprop_lmem(self,out_err,params=None):
        '''
        A limited-memory version of backprop which uses
        a for loop instead of a memory-intensive vectorized
        implementation.
        '''
        if None is params:
            params = self._netp

        errsignal = np.zeros((len(out_err),self.hdim))
        for i,e in enumerate(out_err):
            errsignal[i] = (params.w2 * e).sum(1)
        return errsignal

    def _backprop(self,out_err,params=None):
        '''
        Compute the error of the hidden layer
        by performing backpropagation.

        out_err is the error of the output
        layer for every training example.
        rows are errors.
        '''
        if None is params:
            params = self._netp
        # the dot product of the layer 2 weights with output
        # error for each training example

        try:
            tiled = np.tile(params.w2[np.newaxis,:,:],(out_err.shape[0],1,1))
            return (tiled * out_err[:,np.newaxis,:]).sum(2)
        except ValueError:
            print 'switching to limited memory backprop'
            self._backprop = self._backprop_lmem
            return self._backprop_lmem(out_err,params=params)

        

    def _update(self,inp,wg2,wg1,bg2,bg1):
        if self._include_weight_decay:
            wg2 += self._netp.w2 * self._lambda
            wg1 += self._netp.w1 * self._lambda

        self._netp.w2 -= self._alpha * wg2
        self._netp.w1 -= self._alpha * wg1
        self._netp.b2 -= self._alpha * bg2
        self._netp.b1 -= self._alpha * bg1

        

    def _check_grad(self,params,inp,grad,epsilon = 10**-4):

        def rcst(x):
            return self._rcost(inp,params.roll(x,self.indim,self.hdim))

        def rcstprime(x):
            return self._sparse_ae_cost_unrolled(inp,params.roll(x,self.indim,self.hdim))

        err = check_grad(rcst,rcstprime,params.unroll())

        rc = self._rcost
        e = epsilon
        tolerance = e ** 2
        
        theta = params.unroll()
        num_grad = np.zeros(theta.shape)


        # compute the numerical gradient of the function by
        # varying the parameters one by one.
        for i in range(len(theta)):
            plus = np.copy(theta)
            minus = np.copy(theta)
            plus[i] += e
            minus[i] -= e
            pp = params.roll(plus,self.indim,self.hdim)
            mp = params.roll(minus,self.indim,self.hdim)
            num_grad[i] =  (rc(inp,pp) - rc(inp,mp)) / (2. * e)

        # the analytical gradient
        agp = params.roll(grad,self.indim,self.hdim)
        # the numerical gradient
        ngp = params.roll(num_grad,self.indim,self.hdim)

        diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
        # layer 1 weights difference
        print np.linalg.norm(ngp.w1 - agp.w1) / np.linalg.norm(ngp.w1 + agp.w1)
        # layer 2 weights difference
        print np.linalg.norm(ngp.w2 - agp.w2) / np.linalg.norm(ngp.w2 + agp.w2)
        # layer 1 bias difference
        print np.linalg.norm(ngp.b1 - agp.b1) / np.linalg.norm(ngp.b1 + agp.b1)
        # layer 2 bias difference
        print np.linalg.norm(ngp.b2 - agp.b2) / np.linalg.norm(ngp.b2 + agp.b2)
        print 'Difference between analytical and numerical gradients is %s' % str(diff)
        print 'scipy.optimize error is %s' % str(err)
        print '2norm is %s' % str(np.linalg.norm(num_grad - grad))
        print np.all(np.sign(num_grad) == np.sign(grad))
        #print diff < tolerance
        print '================================================================'
        print ngp.w1.max()
        print agp.w1.max()
        print ngp.w2.max()
        print agp.w2.max()

        '''
        plt.gray()
        plt.figure()
        # plot the analytical gradients on
        # the first row
        plt.subplot(2,4,1)
        plt.imshow(agp.w1)
        plt.subplot(2,4,2)
        plt.imshow(agp.w2)
        plt.subplot(2,4,3)
        plt.plot(agp.b1)
        plt.subplot(2,4,4)
        plt.plot(agp.b2)
        # plot the numerical (brute force)
        # gradients on the second row
        plt.subplot(2,4,5)
        plt.imshow(ngp.w1)
        plt.subplot(2,4,6)
        plt.imshow(ngp.w2)
        plt.subplot(2,4,7)
        plt.plot(ngp.b1)
        plt.subplot(2,4,8)
        plt.plot(ngp.b2)
        plt.show()
        '''
        return ngp.unroll()
        

    def train_minibatch(self,
                        samples,
                        batch_size,
                        epochs,
                        base_filename,
                        progressive = False):

        try:
            if len(samples) % batch_size:
                raise ValueError('samples must be evenly divisible by batch_size')

            nbatches = len(samples) / batch_size
            if nbatches == 1:
                # pass all the samples to train at once
                theta = self.train(samples,epochs,with_bfgs=True,filename=base_filename)
            else:
                # the samples have been split into batches to avoid MemoryErrors
                samples = samples.reshape((nbatches,batch_size,samples.shape[1]))
                for e in range(epochs):
                    for b,batch in enumerate(samples):
                        print 'Epoch %i, Batch %i' % (e,b)
                        fn = base_filename
                        if fn and progressive:
                            fn = '%s_batch_%i.dat' % (base_filename,i)                
                
                        theta = self._train(batch,1,with_bfgs=True,filename=fn)
                        self._netp = Params.roll(theta,self.indim,self.hdim)

        except KeyboardInterrupt:
            pass

        self._netp = Params.roll(theta,self.indim,self.hdim)
        with open('%s.dat' % base_filename,'wb') as f:
            cPickle.dump(self._netp,f)

        return self._netp.unroll()


    def train(self,data,stopping_condition):
        '''
        '''
        # TODO: Group into batches!
        epoch = 0
        error = 99999
        nbatches = len(data)
        while not stopping_condition(epoch,error):
            batch = 0
            while batch < nbatches and not stopping_condition(epoch,error):
                theta,error = self._train(data[batch], 1, with_bfgs = True)
                self._netp = Params.roll(theta,self.indim,self.hdim)
                batch += 1
            epoch += 1 

    def _train(self,inp,iterations,with_bfgs = False,grad_check_freq = None,filename=None):
        

        def rcst(x):
            v = self._rcost(inp,Params.roll(x,self.indim,self.hdim))
            print 'rcst says: %s' % str(v)
            return v

        def rcstprime(x):
            return self._sparse_ae_cost_unrolled(inp,Params.roll(x,self.indim,self.hdim))


        if with_bfgs:
            x0 = self._netp.unroll()
            mn,val,d = bfgs(rcst,
                            x0,
                            fprime = rcstprime,
                            factr=100,
                            maxfun=iterations,
                            disp=1)
            print val
            print d['task']
            print d['warnflag']
            print d['grad'].sum()
            print d['funcalls']
            if filename:
                with open(filename,'wb') as f:
                    cPickle.dump(Params.roll(mn,self.indim,self.hdim),f)
            return mn,val
        else:
            for i in range(iterations):
                if grad_check_freq:
                    gc = i and not i % grad_check_freq
                else:
                    gc = False
                grads = self._sparse_ae_cost(inp,self._netp, check_grad = gc)
                # leave out cost, the first item
                # in the grads tuple
                self._update(inp,*grads[1:])
            return self._netp.unroll()

    def do_reconstruction(self,fetch):
        samples = fetch(10)
        inp,a,z2,z3,recon = self._activate(samples)
        plt.figure()
        for i in range(10):
            plt.subplot(10,1,i + 1)
            plt.plot(samples[i],'r-')
            plt.plot(recon[i],'b-')
        plt.show()


import unittest
class autoencoder_tests(unittest.TestCase):

    def test_backprop_methods_equivalent(self):
        '''
        There are two backprop methods available, a
        completely vectorized implementation, and a limited
        memory one that uses a for-loop. Ensure that their
        output is identical
        '''
        indim = 100
        hdim = 33
        ae = Autoencoder(Params(indim,hdim))
        err = np.random.random_sample((637,indim))
        b = ae._backprop(err)
        blm = ae._backprop_lmem(err)
        self.assertTrue(b.shape == blm.shape)
        self.assertTrue(np.allclose(b,blm))

    def test_weightgrad_methods_equivalent(self):
        '''
        There are two available methods to compute the
        weight gradient: one a memory-intensive, fully-vectorized
        implementation, and the other a method using a for-loop.
        Ensure that the output of the two methods is identical
        '''
        nsamples = 25
        indim = 100
        hdim = 33
        ae = Autoencoder(Params(indim,hdim))
        cost = np.random.random_sample((nsamples,indim))
        activations = np.random.random_sample((nsamples,hdim))
        wg = ae._weight_grad(cost,activations,ae._netp.w2)
        wglm = ae._weight_grad_lmem(cost,activations,ae._netp.w2)
        self.assertTrue(wg.shape == wglm.shape)
        self.assertTrue(np.allclose(wg,wglm))

import argparse
if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Attempted autoencoder implementation')
    aa = parser.add_argument


    # required args
    aa('--indim',
       help='the size of the inputs',
       type=int)
    aa('--hdim',
       help='the size of the hidden layer',
       type=int)
    aa('--fetch',
       help='the name of a module that contains a function called fetch(batch_size).  \
This should fetch minibatches, as well as do any necessary preprocessing of the data',
       type=str)
    aa('--nsamples',
       help = 'the total number of samples to train on',
       type=int)
    aa('--batchsize',
       help='the number of training example in each batch',
       type=int)
    aa('--epochs',
       help='the total number of epochs to train',
       type=int)
    aa('--paramfile',
       help='the base name of files in which parameters will be saved')

    # optional args
    aa('--corrupt',
       help='approx percentage of each input to corrupt (i.e., set to zero)',
       type=float,
       default = 0.0)
    aa('--linear',
       help='use a linear decoder for the output layer',
       default=False,
       action='store_true')
    aa('--guess',
       help='file with pickled initial parameter guess.  \
 This might be the output of a previous training session.  \
If not specified, start from a random guess.',
       type=str,
       default=None)

    
    args = parser.parse_args()


    # size of input and output layers
    n_dim = args.indim
    # size of hidden layer
    h_dim = args.hdim


    if args.guess:
        parms = Params.from_file(args.guess)
    else:
        parms = Params(n_dim,h_dim)
    ae = Autoencoder(parms,args.corrupt,linear_decoder=args.linear)
    fetch = __import__(args.fetch)
    theta = ae.train_minibatch(fetch.fetch(args.nsamples),
                               args.batchsize,
                               args.epochs,
                               args.paramfile)

                       


    
    

    
    




    
