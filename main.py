# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
import imageio
from torchvision.utils import make_grid, save_image

from tqdm import tqdm
import numpy as np

matplotlib.style.use('ggplot')
if torch.backends.mps.is_available():
    device = "mps"
    
else:
    device = "cpu"

#device = "cpu"
print(device)

lr = 0.0002
num_epochs = 50

batch_size = 128

transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,)),
])

train_data = datasets.MNIST(
    root = './data',
    train = True,                         
    transform = transform, 
    download = True,            
)
# test_data = datasets.MNIST(
#     root = './data', 
#     train = False, 
#     transform = transforms.ToTensor()
# )
    

# train_dataset = datasets.ImageFolder(
#     root='afhq/train',
#     transform=train_transform)

# test_dataset = datasets.ImageFolder(
#     root='afhq/val',
#     transform=valid_transform)
# Data Loader (Input Pipeline)

train_data = datasets.MNIST(
    root='../input/data',
    train=True,
    download=True,
    transform=transform
)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

to_pil_image = transforms.ToPILImage()
# train_loader = DataLoader(
#     train_data, batch_size=batch_size, shuffle=True)

# train_loader = DataLoader(
#     train_dataset, batch_size=batch_size, shuffle=True
# )



#train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


z = 128
k = 1 #number of steps to apply to the discriminator
discriminator = nn.Sequential(
                            nn.Linear(784, 1024),
                            nn.LeakyReLU(0.2),
                            nn.Dropout(0.3),
                            nn.Linear(1024, 512),
                            nn.LeakyReLU(0.2),
                            nn.Dropout(0.3),
                            nn.Linear(512, 256),
                            nn.LeakyReLU(0.2),
                            nn.Dropout(0.3),
                            nn.Linear(256, 1),
                            nn.Sigmoid()).to(device)

generator = nn.Sequential(
                        nn.Linear(z, 256),
                        nn.LeakyReLU(0.2),
                        nn.Linear(256, 512),
                        nn.LeakyReLU(0.2),
                        nn.Linear(512, 1024),
                        nn.LeakyReLU(0.2),
                        nn.Linear(1024, 784),
                        nn.Unflatten(1, (1,28,28)),
                        nn.Tanh()).to(device)
                        #nn.Unflatten(1, (3, 128, 128)),
                        #nn.Sigmoid()).to(device)




def generate_images():
    with torch.no_grad():
        z = torch.randn(64, 100).to(device)
        output = generator(z)
        generated_images = output.reshape(64, 3, 128, 128)
        return generated_images


#shows neural net works with each other
#rand = train_dataset[0][0].to(device)
#print(discriminator(rand[None, :, :]))


#gen_images = generate_images()
#print(len(gen_images))
#print(discriminator(gen_images).shape)



#time to train them
def train(generator, discriminator, batch_size, epochs=200, lf=nn.BCELoss(), lr = 3e-4, device = device, z = z):
    beta1 = 0.5
    gen_opt = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    disc_opt = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    #real_targets = torch.ones(batch_size).to(device)
    #fake_targets = torch.zeros(batch_size).to(device)

    
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    s_imgs = []
    print("Training Starting...........")
    
    for epoch in range(epochs):
        for i, (images, label) in enumerate(train_loader):
            ############### (1) Training The Discriminator #####################
            #maximize log(D(x)) + log(1 - D(G(z)))
            batch_size = len(label)
            real_targets = torch.ones(batch_size).to(device)
            fake_targets = torch.zeros(batch_size).to(device)
            #clear the gradient
            discriminator.zero_grad()

            #put image to gpu
            output = discriminator(images.to(device).view(-1, 784)).view(-1)
            
            #compute loss on real images
            realImg_loss = lf(output, real_targets)
            
            #computes gradients
            realImg_loss.backward(retain_graph = True)
            
            D_x = output.mean().item()
            
            #generate noise for generator input vector
            noise = torch.randn(batch_size, z, device=device)
            
            #fake generated images from our generator
            fake_imgs = generator(noise)
            
            #pass into discriminator
            output = discriminator(fake_imgs.view(-1, 784)).view(-1)
            
            #calculate loss
            fakeImg_loss = lf(output, fake_targets)
            
            #compute gradients
            fakeImg_loss.backward(retain_graph = True)
            
            D_G_z1 = output.mean().item()
            
            errD = realImg_loss + fakeImg_loss

            disc_opt.step()
            ############################ That was one training batch for the Discriminator #########################
            
            ############################ Train Generator: maximize log(D(G(z)))##########################################
            
            #clear gradient
            generator.zero_grad()
            
            
            output = discriminator(fake_imgs.view(-1, 784)).view(-1)
            
            
            
            gen_loss = lf(output, real_targets) #basically, how far were off from fooling the discriminator,
            #since we are generating fake images that we want the disc to predict 1 on
        
            #compute gradients
            gen_loss.backward()
            
            D_G_z2 = output.mean().item()
            
            #update generators weights
            gen_opt.step()
            
            
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tDiscriminator Loss: %.4f\tGenerator Loss: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(train_loader),
                         errD.item(), gen_loss.item(), D_x, D_G_z1, D_G_z2))
                
        
            # Save Losses for plotting later
            G_losses.append(gen_loss.item())
            D_losses.append(errD.item())

        fake_imgs = fake_imgs.cpu().detach()
        generated_img = make_grid(fake_imgs)
        save_image(generated_img, f"./fake_img{epoch}.png")
        s_imgs.append(generated_img)

        imgs = [np.array(to_pil_image(img)) for img in s_imgs]
        imageio.mimsave('./generator_images.gif', imgs)

            

            

train(generator, discriminator, batch_size)


torch.save(generator.state_dict(), "./generator.pt")
torch.save(discriminator.state_dict(), "./discriminator.pt")