from data import StandardImgData
from mnist_models import Discriminator, Generator, weights_init
from sgan import SGAN_Manifold_Reg

dataset = 'mnist'
num_classes = 10
latent_dim = 100
batch_size = 128
samples_per_class = 100

img_data = StandardImgData(samples_per_class, batch_size, dataset)

G = Generator(latent_dim).apply(weights_init)
D = Discriminator(num_classes).apply(weights_init)

data_loaders = img_data.get_dataloaders(dataset)
sgan = SGAN_Manifold_Reg(batch_size, latent_dim, num_classes, G, D, data_loaders)
sgan.train(num_epochs=200)
# sgan.eval(epoch_idx=2)
