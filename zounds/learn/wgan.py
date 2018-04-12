from trainer import Trainer
import numpy as np
import torch
from torch.autograd import Variable


class WassersteinGanTrainer(Trainer):
    """
    Args:
        network (nn.Module): the network to train
        latent_dimension (tuple): A tuple that defines the shape of the latent
            dimension (noise) that is the generator's input
        n_critic_iterations (int): The number of minibatches the critic sees
            for every minibatch the generator sees
        epochs: The total number of passes over the training set
        batch_size: The size of a minibatch
        preprocess_minibatch (function): function that takes the current
            epoch, and a minibatch, and mutates the minibatch
        kwargs_factory (callable): function that takes the current epoch and
            outputs args to pass to the generator and discriminator
    """

    def __init__(
            self,
            network,
            latent_dimension,
            n_critic_iterations,
            epochs,
            batch_size,
            preprocess_minibatch=None,
            kwargs_factory=None,
            debug_gradient=False,
            checkpoint_epochs=1):

        super(WassersteinGanTrainer, self).__init__(epochs, batch_size)
        self.checkpoint_epochs = checkpoint_epochs
        self.debug_gradient = debug_gradient
        self.arg_maker = kwargs_factory
        self.preprocess = preprocess_minibatch
        self.n_critic_iterations = n_critic_iterations
        self.latent_dimension = latent_dimension
        self.network = network
        self.critic = network.discriminator
        self.generator = network.generator
        self.samples = None
        self.register_batch_complete_callback(self._log)
        self.generator_optim = None
        self.critic_optim = None

    def _log(self, *args, **kwargs):
        if kwargs['batch'] % 10:
            return
        msg = 'Epoch {epoch}, batch {batch}, generator {generator_score}, ' \
              'real {real_score}, critic {critic_loss}'
        print msg.format(**kwargs)

    def _minibatch(self, data):
        indices = np.random.randint(0, len(data), self.batch_size)
        return data[indices, ...]

    def _gradient_penalty(self, real_samples, fake_samples, kwargs):
        """
        Compute the norm of the gradients for each sample in a batch, and
        penalize anything on either side of unit norm
        """
        import torch
        from torch.autograd import Variable, grad

        real_samples = real_samples.view(fake_samples.shape)

        subset_size = real_samples.shape[0]

        real_samples = real_samples[:subset_size]
        fake_samples = fake_samples[:subset_size]

        alpha = torch.rand(subset_size)
        if self.use_cuda:
            alpha = alpha.cuda()
        alpha = alpha.view((-1,) + ((1,) * (real_samples.dim() - 1)))

        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
        interpolates = Variable(interpolates, requires_grad=True)
        if self.use_cuda:
            interpolates = interpolates.cuda()

        d_output = self.critic(interpolates, **kwargs)

        grad_ouputs = torch.ones(d_output.size())
        if self.use_cuda:
            grad_ouputs = grad_ouputs.cuda()

        gradients = grad(
            outputs=d_output,
            inputs=interpolates,
            grad_outputs=grad_ouputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

    def freeze_generator(self):
        for p in self.generator.parameters():
            p.requires_grad = False

    def unfreeze_generator(self):
        for p in self.generator.parameters():
            p.requires_grad = True

    def freeze_discriminator(self):
        for p in self.critic.parameters():
            p.requires_grad = False

    def unfreeze_discriminator(self):
        for p in self.critic.parameters():
            p.requires_grad = True

    def _debug_network_gradient(self, network):
        if not self.debug_gradient:
            return

        for n, p in network.named_parameters():
            g = p.grad
            if g is not None:
                print(n, g.min().data[0], g.max().data[0], g.mean().data[0])

    def zero_generator_gradients(self):
        self._debug_network_gradient(self.generator)
        self.generator.zero_grad()

    def zero_discriminator_gradients(self):
        self._debug_network_gradient(self.critic)
        self.critic.zero_grad()

    def _init_optimizers(self):
        if self.generator_optim is None or self.critic_optim is None:
            from torch.optim import Adam
            trainable_generator_params = (
                p for p in self.generator.parameters() if p.requires_grad)
            trainable_critic_params = (
                p for p in self.critic.parameters() if p.requires_grad)

            self.generator_optim = Adam(
                trainable_generator_params, lr=0.0001, betas=(0, 0.9))
            self.critic_optim = Adam(
                trainable_critic_params, lr=0.0001, betas=(0, 0.9))

    def _cuda(self, device=None):
        self.generator = self.generator.cuda()
        self.critic = self.critic.cuda()

    def train(self, data):

        self.network.train()
        self.unfreeze_discriminator()
        self.unfreeze_generator()

        data = data.astype(np.float32)

        noise_shape = (self.batch_size,) + self.latent_dimension
        noise = self._tensor(noise_shape)

        self._init_optimizers()

        start = self._current_epoch
        stop = self._current_epoch + self.checkpoint_epochs

        for epoch in xrange(start, stop):
            if epoch >= self.epochs:
                break

            if self.arg_maker:
                kwargs = self.arg_maker(epoch)
            else:
                kwargs = dict()

            for i in xrange(0, len(data), self.batch_size):
                self.zero_generator_gradients()
                self.zero_discriminator_gradients()

                self.freeze_generator()
                self.unfreeze_discriminator()

                for c in xrange(self.n_critic_iterations):

                    self.zero_discriminator_gradients()

                    input_v = self._variable(self._minibatch(data))

                    if self.preprocess:
                        input_v = self.preprocess(epoch, input_v)

                    d_real = self.critic.forward(input_v, **kwargs)

                    # train discriminator on fake data
                    noise.normal_(0, 1)
                    noise_v = Variable(noise, volatile=True)
                    fake = Variable(
                        self.generator.forward(noise_v, **kwargs).data)

                    if self.preprocess:
                        fake = self.preprocess(epoch, fake)

                    d_fake = self.critic.forward(fake, **kwargs)

                    real_mean = torch.mean(d_real)
                    fake_mean = torch.mean(d_fake)
                    gp = self._gradient_penalty(input_v.data, fake.data, kwargs)
                    d_loss = (fake_mean - real_mean) + gp
                    d_loss.backward()
                    self.critic_optim.step()

                self.zero_discriminator_gradients()
                self.zero_generator_gradients()

                self.unfreeze_generator()
                self.freeze_discriminator()

                # train generator
                noise.normal_(0, 1)
                noise_v = Variable(noise)
                fake = self.generator.forward(noise_v, **kwargs)

                if self.preprocess:
                    fake = self.preprocess(epoch, fake)

                self.samples = fake

                d_fake = self.critic.forward(fake, **kwargs)
                g_loss = -torch.mean(d_fake)
                g_loss.backward()
                self.generator_optim.step()

                gl = g_loss.data[0]
                dl = d_loss.data[0]
                rl = real_mean.data[0]

                self.on_batch_complete(
                    epoch=epoch,
                    batch=i,
                    generator_score=gl,
                    real_score=rl,
                    critic_loss=dl,
                    samples=self.samples,
                    network=self.network)

            self._current_epoch += 1

        return self.network
