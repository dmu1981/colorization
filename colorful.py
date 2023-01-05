"""Color classification [colorful colorization, https://richzhang.github.io/colorization/]"""
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
BATCH_SIZE = 16

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
          Up(  16,  400, nn.LogSoftmax(dim=1)),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x).view(-1,400,IMAGE_SIZE*IMAGE_SIZE)

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
        torchvision.transforms.Lambda(lambda x: (x + 1.0) / 2.0),
        torchvision.transforms.Lambda(lambda x: 0.1 + 0.8*x)
    ])
  
  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    img = self.transform(torchvision.io.read_image(os.path.join(self.path, self.imgs[idx])))
    
    yuv = kornia.color.yuv.rgb_to_yuv(img)

    u = torch.clip(torch.floor((yuv[1,:,:] + 0.5) * 20.0), 0, 19)
    v = torch.clip(torch.round((yuv[2,:,:] + 0.5) * 20.0), 0, 19)
    color_bin = (u * 20 + v)

    return img, yuv[0, :, :].view(1,IMAGE_SIZE,IMAGE_SIZE)*2-1, color_bin.view(1, IMAGE_SIZE, IMAGE_SIZE)

def recover_uv(color_bin):
  u = torch.div(color_bin, 20, rounding_mode='floor') / 20.0 - 0.5
  v = torch.remainder(color_bin, 20) / 20.0 - 0.5  
  return u, v

dataloader = DataLoader(ImageDataset("./train/dogs"), batch_size=BATCH_SIZE, shuffle=True)
MODEL_PATH = "colorful.pt"

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

        self.calculate_loss_weights()

    def save_images(self, epoch, original_sample, bw_sample, estimated_bins):
        """Generate some images and save them to disk for review"""
  
        bw_sample=(bw_sample[:8,:,:].view(8,1,IMAGE_SIZE, IMAGE_SIZE)+1)/2


        u, v = recover_uv(estimated_bins)
        print(estimated_bins.size())
        reconstructed_sample=torch.cat([bw_sample, u[:8,:,:], v[:8,:,:]], dim=1)        
        reconstructed_sample=kornia.color.yuv.yuv_to_rgb(reconstructed_sample)
        
        
        bw_sample=torch.concat([bw_sample, bw_sample, bw_sample], dim=1)

        img = torch.concat([original_sample[:8,:,:], bw_sample, reconstructed_sample], dim=0)
        
        grid = torchvision.utils.make_grid(img, nrow=8)
        im = torchvision.transforms.ToPILImage()(grid)
        im.save("colorful_epoch_{}.png".format(epoch))

    def calculate_loss_weights(self):
        try:
          cp = torch.load("colorful_weights.pt")
          self.weights = cp["weights"].to(device)
          return
        except:
          print("Could not load color bin weights, re-calculating")
        
        total = torch.zeros(400)
        # Calculate the histogram of all color bins
        for original, bw, color_bin in tqdm(dataloader):
          hist, bins = torch.histogram(color_bin, bins=400, range=(0, 400))
          total += hist

        # Turn into a distribution
        dist = total / torch.sum(total)

        # Smooth with a gaussian kernel (compare original paper)
        dist = dist.view(1, 20, 20)
        width=9
        sigma=5
        distance = torch.arange(
            -(width // 2), width // 2 + 1, dtype=torch.float, device="cpu"
        )
        gaussian = torch.exp(
            -(distance[:, None] ** 2 + distance[None] ** 2) / (2 * sigma ** 2)
        )
        gaussian /= gaussian.sum()
        kernel = gaussian[None, None].expand(1, -1, -1, -1)
        dist = torch.nn.functional.conv2d(dist, kernel, padding=width//2, groups=1)
        dist = (dist.view(1,1,400) + (1/400)) / 2
        dist = dist.view(400)
        weights = 1.0 / dist
        weights /= weights.sum()

        print(weights)
        torch.save({ 
          "weights": weights 
          }, "colorful_weights.pt")

        self.weights = weights.to(device)

    def train(self):
        """Train some epochs"""
        loss_function = torch.nn.NLLLoss(weight=self.weights)

        for epoch in range(self.start_epoch+1, 25000):
            bar = tqdm(dataloader)
            total_loss = 0
            cnt = 0
            for original_sample, bw_sample, color_bins in bar:              
              bw_sample = bw_sample.to(device)
              color_bins = color_bins.to(device).view(-1,IMAGE_SIZE*IMAGE_SIZE).type(torch.long)
              self.optimizer.zero_grad()
              x = self.model(bw_sample)
              loss = loss_function(x, color_bins)
              loss.backward()
              total_loss += loss.item()
              cnt += 1
              self.optimizer.step()
              bar.set_description("epoch {}, loss {:.8f}".format(epoch, total_loss / cnt*1000))
              
            original_sample, bw_sample, _ = next(iter(dataloader))
            x = self.model(bw_sample.to(device)).view(-1, 400, IMAGE_SIZE, IMAGE_SIZE)
            x = torch.argmax(x, dim=1).view(-1,1, IMAGE_SIZE, IMAGE_SIZE)
            self.save_images(epoch, original_sample.to("cpu"), bw_sample.to("cpu"), x.to("cpu"))

            torch.save({
                      'epoch': epoch,
                      'model_state_dict': 
                        self.model.state_dict(),
                      'optimizer_state_dict': 
                        self.optimizer.state_dict(),
                      }, MODEL_PATH)

trainer = Trainer()
trainer.train()
  