"""L1-loss based colorization network"""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
from PIL import Image
import torchvision
from tqdm import tqdm
import kornia

device = "cuda"
LATENT_DIM = 100
IMAGE_SIZE = 128
BATCH_SIZE = 256

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fw = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(5,5),
                      padding=2),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(5,5),
                      stride=(2,2),
                      padding=2),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.2)
          )
        
    def forward(self, x):
        return self.fw(x)

class Up(nn.Module):
    """Strided up convolution, resolution is doubled"""
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU(0.2)):
        super().__init__()
        self.fw = nn.Sequential(
          nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(num_features=out_channels),
          nn.LeakyReLU(0.2),
          nn.Conv2d(in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=(5,5),
                    padding=2),
          nn.BatchNorm2d(num_features=out_channels),
          activation,
        )

    def forward(self, x):
        return self.fw(x)

class Colorizer(nn.Module):
    """The colorizer"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
          Down(   1,   8),
          Down(   8,  32),
          Down(  32, 128),
          Down( 128, 512),
        )
        self.decoder = nn.Sequential(
          Up( 512, 128),
          Up( 128,  32),
          Up(  32,  16),
          Up(  16,  2, nn.Tanh()),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

class ImageDataset(Dataset):
  def __init__(self, path):
    self.path = path
    self.imgs = os.listdir(path)

    self.transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMAGE_SIZE,IMAGE_SIZE),
          interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.Lambda(lambda x: x.type(torch.float)),
        torchvision.transforms.Normalize(128, 128),
        # Avoid 100% black or white as that will create ugly artifacts during color space transformation
        torchvision.transforms.Lambda(lambda x: torch.tanh(x)*1.25),         
        torchvision.transforms.Lambda(lambda x: (x + 1.0) / 2.0),
        torchvision.transforms.Lambda(lambda x: x.to(device))
    ])
  
  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    img = self.transform(torchvision.io.read_image(os.path.join(self.path, self.imgs[idx])))
    
    yuv = kornia.color.yuv.rgb_to_yuv(img)

    return yuv[0, :, :].view(1,IMAGE_SIZE,IMAGE_SIZE)*2-1, yuv[1:3, :, :]*20

dataloader = DataLoader(ImageDataset("./train/dogs"), batch_size=BATCH_SIZE, shuffle=True)
MODEL_PATH = "colorize.pt"

class Trainer:
    """ The training class for the colorizer network"""
    def __init__(self):
        self.model = Colorizer().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        self.start_epoch = 0
        try:
            checkpoint = torch.load(MODEL_PATH)

            self.model.\
              load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.\
              load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.start_epoch = checkpoint['epoch']
        except:
            print("Could not load model from disk, starting from scratch")

    def save_images(self, epoch, color_sample, bw_sample, reconstructed_sample):
        """Generate some images and save them to disk for review"""
  
        bw_sample=(bw_sample[:8,:,:].view(8,1,IMAGE_SIZE, IMAGE_SIZE)+1)/2

        color_sample=torch.cat([bw_sample, color_sample[:8,:,:,:]/20], dim=1)
        color_sample=kornia.color.yuv.yuv_to_rgb(color_sample)

        reconstructed_sample=torch.cat([bw_sample, reconstructed_sample[:8,:,:,:]/20], dim=1)        
        reconstructed_sample=kornia.color.yuv.yuv_to_rgb(reconstructed_sample)
        
        
        bw_sample=torch.concat([bw_sample, bw_sample, bw_sample], dim=1)

        img = torch.concat([color_sample, bw_sample, reconstructed_sample], dim=0)
        
        grid = torchvision.utils.make_grid(img, nrow=8)
        im = torchvision.transforms.ToPILImage()(grid)
        im.save("epoch_{}.png".format(epoch))

    def train(self):
        """Train some epochs"""
        loss_function = torch.nn.L1Loss()

        for epoch in range(self.start_epoch+1, 25000):
            bar = tqdm(dataloader)
            total_loss = 0
            cnt = 0
            for bw_sample, color_sample in bar:
              bw_sample = bw_sample.to(device)
              color_sample = color_sample.to(device)
              self.optimizer.zero_grad()
              x = self.model(bw_sample)
              loss = loss_function(x, color_sample)
              loss.backward()
              total_loss += loss.item()
              cnt += 1
              self.optimizer.step()
              bar.set_description("epoch {}, loss {:.8f}".format(epoch, total_loss / cnt*1000))
              
            self.save_images(epoch, color_sample, bw_sample, x)

            torch.save({
                      'epoch': epoch,
                      'model_state_dict': 
                        self.model.state_dict(),
                      'optimizer_state_dict': 
                        self.optimizer.state_dict(),
                      }, MODEL_PATH)

if __name__ == "__main__":
  Trainer = Trainer()
  Trainer.train()
