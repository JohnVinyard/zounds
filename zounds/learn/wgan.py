from trainer import Trainer
import numpy as np


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
        on_batch_complete (callable): callable invoked after each epoch,
            accepting epoch and network being trained as arguments
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
            on_batch_complete=None):

        super(WassersteinGanTrainer, self).__init__(epochs, batch_size)
        self.on_batch_complete = on_batch_complete
        self.arg_maker = kwargs_factory
        self.preprocess = preprocess_minibatch
        self.n_critic_iterations = n_critic_iterations
        self.latent_dimension = latent_dimension
        self.network = network
        self.critic = network.discriminator
        self.generator = network.generator

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

        # computing the norm of the gradients is very expensive, so I'm only
        # taking a subset of the minibatch here
        subset_size = 10

        real_samples = real_samples[:subset_size]
        fake_samples = fake_samples[:subset_size]

        # TODO: this should have the same number of dimensions as real and
        # fake samples, and should not be hard-coded
        alpha = torch.rand(subset_size).cuda()
        alpha = alpha.view((-1,) + ((1,) * (real_samples.dim() - 1)))

        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
        interpolates = Variable(interpolates.cuda(), requires_grad=True)

        d_output = self.critic(interpolates, **kwargs)

        gradients = grad(
            outputs=d_output,
            inputs=interpolates,
            grad_outputs=torch.ones(d_output.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

    def train(self, data):

        import torch
        from torch.optim import Adam
        from torch.autograd import Variable

        data = data.astype(np.float32)

        zdim = self.latent_dimension

        noise_shape = (self.batch_size,) + self.latent_dimension
        noise = torch.FloatTensor(*noise_shape)
        fixed_noise = torch.FloatTensor(*noise_shape).normal_(0, 1)

        self.generator.cuda()
        self.critic.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        trainable_generator_params = (
            p for p in self.generator.parameters() if p.requires_grad)
        trainable_critic_params = (
            p for p in self.critic.parameters() if p.requires_grad)

        generator_optim = Adam(
            trainable_generator_params, lr=0.0001, betas=(0, 0.9))
        critic_optim = Adam(
            trainable_critic_params, lr=0.0001, betas=(0, 0.9))

        for epoch in xrange(self.epochs):
            if self.arg_maker:
                kwargs = self.arg_maker(epoch)
            else:
                kwargs = dict()

            for i in xrange(0, len(data), self.batch_size):
                self.generator.zero_grad()
                self.critic.zero_grad()

                self.generator.eval()
                self.critic.train()

                for c in xrange(self.n_critic_iterations):

                    self.critic.zero_grad()

                    minibatch = self._minibatch(data)
                    inp = torch.from_numpy(minibatch)
                    inp = inp.cuda()
                    input_v = Variable(inp)

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
                    d_loss = \
                        (fake_mean - real_mean) \
                        + self._gradient_penalty(input_v.data, fake.data, kwargs)
                    d_loss.backward()
                    critic_optim.step()

                self.generator.train()
                self.critic.eval()

                # train generator
                noise.normal_(0, 1)
                noise_v = Variable(noise)
                fake = self.generator.forward(noise_v, **kwargs)

                if self.preprocess:
                    fake = self.preprocess(epoch, fake)

                d_fake = self.critic.forward(fake, **kwargs)
                g_loss = -torch.mean(d_fake)
                g_loss.backward()
                generator_optim.step()

                gl = g_loss.data[0]
                dl = d_loss.data[0]

                if self.on_batch_complete:
                    self.on_batch_complete(epoch, self.network)

                if i % 10 == 0:
                    print \
                        'Epoch {epoch}, batch {i}, generator {gl}, critic {dl}' \
                            .format(**locals())

        return self.network