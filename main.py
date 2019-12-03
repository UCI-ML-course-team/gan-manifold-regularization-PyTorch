from data import StandardImgData
from models import Discriminator, Generator, weights_init
from sgan import SGAN_Manifold_Reg

dataset = 'mnist'
num_classes = 10
latent_dim = 100
batch_size = 256
samples_per_class = 100
unlb_samples_per_class = 5000

img_data = StandardImgData(samples_per_class, batch_size, dataset, unlb_samples_per_class)

G = Generator(latent_dim).apply(weights_init)
D = Discriminator(num_classes).apply(weights_init)

data_loaders = img_data.get_mnist_dataloaders()
sgan = SGAN_Manifold_Reg(batch_size, latent_dim, num_classes, G, D, data_loaders)
sgan.train(num_epochs=20)
# ssl.eval(epoch_idx=2)
