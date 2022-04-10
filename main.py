import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import make_grid
# from torch.utils.tensorboard import SummaryWriter
from network import Generator, Discriminator
from fmnist import fmnist
from plotting import real_image_per_class_grid

def train(lr, batch_size, num_epochs, device, data_dir, save_dir, num_workers, img_dim, noise_dim):
    """
    Returns Plot for Real and Generated Images side by side for each class.
    Inputs:
        lr - Learning Rate
        batch_size - Batch size to use for the data loaders
        num_epochs - Number of epochs to train the model for.
        device - Device to use for training.
        data_dir - Directory in which the FashionMNIST dataset should be downloaded.
        save_dir - Directory in which the model should be saved.
        num_workers - Number of workers to use in the data loaders.
        img_dim - Dimension of the images(28 * 28).
        noise_dim - Dimension of the noise vector.
    """

    # Create a summary writer for TensorBoard and create a directory 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_name = os.path.join('./Sensity_Project_Results/CGAN/')
    tfb_name_train = os.path.join(checkpoint_name + '/train')
    if not os.path.exists(tfb_name_train):
        os.makedirs(tfb_name_train)
    # writer = SummaryWriter(tfb_name_train)

    # Load the dataset
    train_loader = fmnist(data_dir, batch_size, num_workers)

    # Load the discriminator and generator
    netD = Discriminator(img_dim, output_dim = 1)
    netG = Generator(img_dim, noise_dim)

    # Move the models to the desired device
    netD.to(device)
    netG.to(device)

    # Define the loss function
    criterion = nn.BCELoss()

    # Define the optimizers for discriminator and generator
    optimizer_D = torch.optim.Adam(netD.parameters(), lr = lr)
    optimizer_G = torch.optim.Adam(netG.parameters(), lr = lr)
    
    # Define empty lists to append discriminator and generator losses per epoch
    loss_G = []
    loss_D = []

    # Load model
    if os.path.exists('./Sensity_Project_Results/CGAN/netG.pth'):
        netG.load_state_dict(torch.load('./Sensity_Project_Results/CGAN/netG.pth'))

    if os.path.exists('/Sensity_Project_Results/CGAN/netD.pth'):
        netD.load_state_dict(torch.load('./Sensity_Project_Results/CGAN/netD.pth'))

    # Training 
    for epoch in range(num_epochs):
        running_lossD = 0.0
        running_lossG = 0.0
        for i, (inputs, labels) in tqdm(enumerate(train_loader)):
            # Train Discriminator with real images
            inputs = inputs.view(batch_size, img_dim).to(device) 
            labels = labels.to(device) 
            binary_labels_1 = torch.ones(batch_size).to(device)
            binary_labels_0 = torch.zeros(batch_size).to(device)
            optimizer_D.zero_grad()
            outputs = netD(inputs, labels).view(batch_size)
            lossD_real = criterion(outputs, binary_labels_1)

            # Train Discriminator with fake images
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
            fake_images = netG(noise, fake_labels, noise_dim) 
            fake_outputs = netD(fake_images, fake_labels).view(batch_size)
            lossD_fake = criterion(fake_outputs, binary_labels_0)

            # Discriminator loss
            running_lossD += lossD_real.item() + lossD_fake.item()
            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
            fake_images = netG(noise, fake_labels, noise_dim) 
            fake_outputs = netD(fake_images, fake_labels).view(batch_size)
            lossG = criterion(fake_outputs, binary_labels_1)
            running_lossG += lossG.item()
            lossG.backward()
            optimizer_G.step()

        # Calculation of per epoch loss
        epoch_loss_G = running_lossG / len(train_loader)
        epoch_loss_D = running_lossD / len(train_loader)
        print(f'epoch [{epoch + 1}/{num_epochs}], Discriminator loss:{epoch_loss_D:.4f}, Generator loss:{epoch_loss_G:.4f}')

        # Saving on tensorboard
        # writer.add_scalar('Discriminator Loss', epoch_loss_D, epoch)
        # writer.add_scalar('Generator Loss', epoch_loss_G, epoch)

        # Append the loss values
        loss_D.append(epoch_loss_G)
        loss_G.append(epoch_loss_D)

        # Save the model
        torch.save(netG.state_dict(), './Sensity_Project_Results/CGAN/netG.pth')
        torch.save(netD.state_dict(), './Sensity_Project_Results/CGAN/netD.pth')     
    
    # Loss plot for discriminator and generator
    d_loss, = plt.plot(loss_D, label = 'Discriminator Loss')   
    g_loss, = plt.plot(loss_G, label = 'Generator Loss')   
    plt.legend(loc = 'upper right',  prop = {'size': 20})
    plt.legend(handles = [d_loss, g_loss])
    plt.title('CGAN Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    fname = 'CGAN'  + '.png'
    path = save_dir
    fname = os.path.join(path, fname)
    plt.savefig(fname, bbox_inches = 'tight')
    #plt.show()    

    # Plotting function generated and real image side by side for each class
    # Grid for each class of FashionMNIST
    grid_real = real_image_per_class_grid(data_dir, batch_size, num_workers)

    # Create a grid for generated images
    f_noise = Variable(torch.randn(10, noise_dim)).to(device)
    f_labels = torch.LongTensor([i for i in range(10) for _ in range(1)]).to(device)
    f_images = netG(f_noise, f_labels, noise_dim).cpu().unsqueeze(1)
    grid_gen = make_grid(f_images, nrow = 10, normalize = True)

    # Merge the two grids
    merge_grid = torch.cat((grid_real.permute(1, 2, 0), grid_gen.permute(1, 2, 0)), axis = 0)

    # Plot the merged grid
    _, ax = plt.subplots(figsize=(36 ,24))
    ax.imshow(merge_grid.data, cmap='gray')
    _ = plt.yticks(np.arange(15, 60, 30), ['Real Images', 'Generated Images'], rotation = 0, fontsize = 30)
    _ = plt.xticks(np.arange(15, 300, 30), 
                ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'], 
                rotation = 0, fontsize = 30)
    fig = ax.figure
    path = save_dir
    fname = 'Real_and_Generated_Images_side_by_side' + '.png'
    fname = os.path.join(path, fname)
    fig.savefig(fname, bbox_inches = 'tight')

if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default = 1e-4, type = float,
                        help = 'Learning rate to use')
    parser.add_argument('--batch_size', default = 32, type = int,
                        help = 'Minibatch size')

    # Other hyperparameters
    parser.add_argument('--num_epochs', default = 40, type = int,
                        help = 'Max number of epochs')
    parser.add_argument('--data_dir', default = 'data_dir/', type = str,
                        help = 'Data directory where to store/find the FashionMNIST dataset.')
    parser.add_argument('--save_dir', default = './Sensity_Project_Results/', type = str,
                        help = 'Data directory where to store/find the FashionMNIST dataset.')
    parser.add_argument('--num_workers', default = 0, type = int,
                        help = 'Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0.')   
    parser.add_argument('--img_dim', default = 28 * 28, type = int,
                        help = 'Size of FashionMNIST images which is 784')       
    parser.add_argument('--noise_dim', default = 64, type = int,
                        help = 'Dimension of the noise vector')               

    args = parser.parse_args()
    # Use GPU if available, else use CPU
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    kwargs = vars(args)
    train(**kwargs)