from trainer import Trainer
import numpy as np


class GanTrainer(Trainer):
    def __init__(
            self,
            network,
            loss,
            generator_optim_func,
            discriminator_optim_func,
            latent_dimension,
            epochs,
            batch_size):

        super(GanTrainer, self).__init__(epochs, batch_size)
        self.discriminator_optim_func = discriminator_optim_func
        self.generator_optim_func = generator_optim_func
        self.latent_dimension = latent_dimension
        self.loss = loss
        self.network = network
        self.discriminator = network.discriminator
        self.generator = network.generator

    def train(self, data):

        import torch
        from torch.autograd import Variable

        data = data.astype(np.float32)

        zdim = self.latent_dimension

        noise_shape = (self.batch_size,) + self.latent_dimension
        noise = torch.FloatTensor(*noise_shape)
        fixed_noise = torch.FloatTensor(*noise_shape).normal_(0, 1)
        label = torch.FloatTensor(self.batch_size)
        real_label = 1
        fake_label = 0

        self.generator.cuda()
        self.discriminator.cuda()
        self.generator.train()
        self.discriminator.train()
        self.loss.cuda()
        label = label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        generator_optim = self.generator_optim_func(self.generator)
        discriminator_optim = self.discriminator_optim_func(self.discriminator)

        for epoch in xrange(self.epochs):
            for i in xrange(0, len(data), self.batch_size):
                minibatch = data[i: i + self.batch_size]

                if len(minibatch) != self.batch_size:
                    continue

                inp = torch.from_numpy(minibatch)
                inp = inp.cuda()

                # train discriminator on real data with one-sided label
                # smoothing
                self.discriminator.zero_grad()
                soft_labels = \
                    0.7 + (np.random.random_sample(self.batch_size) * 0.4) \
                        .astype(np.float32)
                soft_labels = torch.from_numpy(soft_labels)
                label.copy_(soft_labels)

                input_v = Variable(inp)
                label_v = Variable(label)

                output = self.discriminator.forward(input_v)
                real_error = self.loss(output, label_v)
                real_error.backward()

                # train discriminator on fake data
                noise.normal_(0, 1)
                noise_v = Variable(noise)
                fake = self.generator.forward(noise_v)
                label.fill_(fake_label)
                label_v = Variable(label)
                output = self.discriminator.forward(fake.detach())
                fake_error = self.loss(output, label_v)
                fake_error.backward()
                discriminator_error = real_error + fake_error
                discriminator_optim.step()

                # train generator
                self.generator.zero_grad()
                label.fill_(real_label)
                label_v = Variable(label)
                output = self.discriminator.forward(fake)
                generator_error = self.loss(output, label_v)
                generator_error.backward()
                generator_optim.step()

                gl = generator_error.data[0]
                dl = discriminator_error.data[0]
                re = real_error.data[0]
                fe = fake_error.data[0]

                if i % 10 == 0:
                    print \
                        'Epoch {epoch}, batch {i}, generator {gl}, real_error {re}, fake_error {fe}' \
                            .format(**locals())

        return self.network
