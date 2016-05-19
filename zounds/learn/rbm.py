import numpy as np
from util import sigmoid, stochastic_binary as sb
import hashlib


class Rbm(object):
    """
    A binary->binary
    `Restricted Boltzmann Machine <http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf>`_.

    Adapted from matlab code written by Geoff Hinton, available
    `here <http://www.cs.toronto.edu/~hinton/code/rbm.m>`_

    Justifications for initialization parameters can be found
    `here <http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf>`_

    TODO: More in-depth explanation of RBMs
    """

    def __init__(
            self,
            indim,
            hdim,
            learning_rate=0.1,
            weight_decay=.002,
            initial_momentum=.5,
            final_momentum=.9,
            sparsity_target=.01,
            sparsity_decay=.9,
            sparsity_cost=.001):

        """__init__

        :param indim: The number of units in the visible layer, i.e., the \
        dimension of training data vectors

        :param hdim: The number of units in the hidden layer, i.e., the \
        dimension of the representation

        :param learning_rate:  The size of "step" to take when adjusting weights \
        during the update step

        :param weight_decay: A value which discourages large weights

        :param initial_momentum: The amount by which momentum from previous \
        updates decays from epoch to epoch, for the first five epochs.

        :param final_momentum: The amount by which momentum from previous \
        updates decays from epoch to epoch, after the first five epochs.

        :param sparsity_target:  The average number of hidden units that should \
        be "on" at once.

        :param sparsity_decay: The amount by which the sparsity penalty decays from \
        epoch to epoch

        :param sparsity_cost: The importance of the sparsity target.  Higher numbers \
        increase the importance of sparsity to the learning objective.
        """

        super(Rbm, self).__init__()

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
        self._weights = np.random.normal(0.0, .01, (self._indim, self._hdim))

        # learning rate, weight decay, momentum
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._initial_momentum = initial_momentum
        self._final_momentum = final_momentum

    @property
    def version(self):
        h = hashlib.md5(self._weights)
        h.update(self._vbias)
        h.update(self._hbias)
        return h.hexdigest()

    @property
    def indim(self):
        """
        The size of the visible layer, i.e., the dimension of input data vectors
        """
        return self._indim

    @property
    def hdim(self):
        """
        The size of the hidden layer, i.e. the dimension of the learned
        representation
        """
        return self._hdim

    @property
    def dim(self):
        return self.hdim

    def _up(self, v):
        """
        Given a visible sample, return the pre-sigmoid and
        sigmoid activations of the hidden units
        """
        pre_sigmoid = np.dot(v, self._weights) + self._hbias
        return pre_sigmoid, sigmoid(pre_sigmoid)

    def _h_from_v(self, v):
        """
        Infer the state of the stochastic binary
        hidden units
        """
        ps, s = self._up(v)
        return ps, s, sb(s)

    def _down(self, h):
        """
        Given a hidden state, infer the activation of
        the visible units.
        """
        pre_sigmoid = np.dot(h, self._weights.T) + self._vbias
        sig = sigmoid(pre_sigmoid)
        return pre_sigmoid, sig, sb(sig)

    def _v_from_h(self, h):
        """
        At the moment, I'm following the pattern of the
        rbm in the theano tutorial, but this is really just
        a superfluous wrapper
        """
        ps, s, stoch = self._down(h)
        return ps, s, stoch

    def _gibbs_hvh(self, h):
        """
        One step of gibbs sampling, starting from the
        hidden layer
        """
        vps, vs, vstoch = self._v_from_h(h)
        ps, s, stoch = self._h_from_v(vs)
        return vs, ps, s, stoch

    def _positive_phase(self, inp):
        """
        positive phase sample (reality)
        get the pre-sigmoid, sigmoid, and stochastic
        binary activations of the hidden units, given
        the visible activations
        """
        ps, s, stoch = self._h_from_v(inp)
        posprod = np.dot(inp.T, s)
        pos_h_act = s.sum(axis=0)
        pos_v_act = inp.sum(axis=0)
        return stoch, posprod, pos_h_act, pos_v_act

    def _negative_phase(self, stoch):
        """
        negative phase sample (fantasy)
        """
        vs, gps, gs, gstoch = self._gibbs_hvh(stoch)
        negprod = np.dot(vs.T, gs)
        neg_h_act = gs.sum(axis=0)
        neg_v_act = vs.sum(axis=0)
        return vs, negprod, neg_h_act, neg_v_act

    def _update(self, inp, momentum, epoch, batch):
        """
        Compare the energy of the input with the energy
        of a fantasy produced by beginning with that input and
        doing one step of gibbs sampling

        Update the weights by multiplying the difference between
        the input and fantasy free energies by a learning rate
        """

        # positive phase (reality)
        stoch, posprod, pos_h_act, pos_v_act = self._positive_phase(inp)

        # negative phase (fantasy)
        v, negprod, neg_h_act, neg_v_act = self._negative_phase(stoch)

        # calculate reconstruction error
        error = np.sum(np.abs(inp - v) ** 2)

        # number of samples in this batch
        n = float(inp.shape[0])
        m = momentum
        lr = self._learning_rate
        wd = self._weight_decay

        # sparsity
        if self._sparsity_target is not None:
            current_sparsity = stoch.sum(0) / float(n)
            self._sparsity = (self._sparsity_decay * self._sparsity) + \
                             ((1 - self._sparsity_decay) * current_sparsity)
            sparse_penalty = self._sparsity_cost * \
                             (self._sparsity_target - self._sparsity)

        # update the weights
        self._wvelocity = (m * self._wvelocity) + \
                          lr * (
                              ((posprod - negprod) / n) - (wd * self._weights))
        self._weights += self._wvelocity
        if self._sparsity_target is not None:
            self._weights += sparse_penalty

        # update the visible biases
        self._vbvelocity = (m * self._vbvelocity) + \
                           ((lr / n) * (pos_v_act - neg_v_act))
        self._vbias += self._vbvelocity

        # update the hidden biases
        self._hbvelocity = (m * self._hbvelocity) + \
                           ((lr / n) * (pos_h_act - neg_h_act))
        self._hbias += self._hbvelocity
        if self._sparsity_target is not None:
            self._hbias += sparse_penalty

        return error

    def train(self, samples, stopping_condition):
        """train

        :param samples: A two-dimensional numpy array whose second dimension \
        should be equal to :py:meth:`Rbm.indim`

        :param stopping_condition: A callable which takes epoch and error as \
        arguments

        """

        batch_size = 100
        nbatches = int(len(samples) / batch_size)
        # If the number of samples isn't evenly divisible by batch size, we're
        # just going to shave off the end.  If the training set is sufficiently
        # large (hopefully in the tens or hundreds of thousands), this shouldn't
        # be a big deal.
        samples = samples[:nbatches * batch_size].reshape(
                (nbatches, batch_size, samples.shape[1]))
        # initialize some variables we'll
        # use during training, but then throw away
        self._wvelocity = np.zeros(self._weights.shape)
        self._vbvelocity = np.zeros(self._indim)
        self._hbvelocity = np.zeros(self._hdim)
        self._sparsity = np.zeros(self._hdim)

        epoch = 0
        error = 99999
        nbatches = len(samples)
        while not stopping_condition(epoch, error):
            if epoch < 5:
                mom = self._initial_momentum
            else:
                mom = self._final_momentum
            batch = 0
            while batch < nbatches and not stopping_condition(epoch, error):
                error = self._update(samples[batch], mom, epoch, batch)
                batch += 1
                print 'Epoch %i, Batch %i, Error %1.4f' % (epoch, batch, error)
            epoch += 1

        # get rid of the "temp" training variables
        del self._wvelocity
        del self._vbvelocity
        del self._hbvelocity
        del self._sparsity

    def activate(self, inp, binarize=True):
        """
        Activate the net using the visible sample.
        Threshold the probabilities that the hidden
        units are on.
        """
        hps, hs, hstoch = self._h_from_v(inp)
        vps, vs, vstoch = self._v_from_h(hs)
        if binarize:
            vs[vs > .5] = 1
            vs[vs <= .5] = 0
        return vs

    def fromfeatures(self, features, binarize=True):
        """
        Using the hidden or latent variables, return the reconstructed output
        """
        vps, vs, vstoch = self._v_from_h(features)
        if binarize:
            vs[vs > .5] = 1
            vs[vs <= .5] = 0
        return vs

    # TODO: Is this the correct implementation for both Rbm and LinearRbm?
    def __call__(self, data, binarize=True):
        """__call__

        :param data: A two-dimensional numpy array of input data vectors

        :returns: The activations of the hidden layer for each input example
        """

        ps, s, stoch = self._h_from_v(data)
        if binarize:
            s[s > .5] = 1
            s[s <= .5] = 0
        return s


class RealValuedRbm(Rbm):
    """
    A Restricted Boltzmann Machine with gaussian visible units, as described
    in section 13.2 "Gaussian visible units",
    `here <http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf>`_.

    When learning a deep-belief network, an instance of this class usually \
    constitutes the first layer, as the raw data is typically real-valued, \
    instead of binary.
    """

    def __init__(self, indim, hdim, sparsity_target=.01, learning_rate=.001):
        """__init__

        :param indim: The number of units in the visible layer, i.e., the \
        dimension of training data vectors

        :param hdim: The number of units in the hidden layer, i.e., the \
        dimension of the representation

        :param sparsity_target:  The average number of hidden units that should \
        be "on" at once.

        :param learning_rate:  The size of "step" to take when adjusting weights \
        during the update step.  The default :code:`learning_rate` is lower than \
        in the :py:class:`Rbm` because *"The learning rate needs to be about one \
        or two orders of magnitude smaller than when using binary visible units \
        ...A smaller learning rate is required because there is no upper bound \
        to the size of a component in the reconstruction and if one component \
        becomes very large, the weights emanating from it will get a very big \
        learning signal".*, from section 13.2 "Gaussian visible units", available \
        `here <http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf>`_

        """
        super(RealValuedRbm, self).__init__(
                indim,
                hdim,
                learning_rate=learning_rate,
                sparsity_target=sparsity_target)

    def _gibbs_hvh(self, h):
        """
        One step of gibbs sampling, starting from the
        hidden layer
        """
        vps, vs, vstoch = self._v_from_h(h)
        # use the actual value of the visible units,
        # instead of the value passed through a sigmoid function
        ps, s, stoch = self._h_from_v(vps)
        return vps, ps, s, stoch

    def activate(self, inp):
        hps, hs, hstoch = self._h_from_v(inp)
        vps, vs, vstoch = self._v_from_h(hs)
        return vps
