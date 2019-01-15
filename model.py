import torch
import torch.nn as nn

class ColorizerV1(nn.Module):
    def __init__(self):
        super(ColorizerV1, self).__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256), 
            nn.ReLU(),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 2, kernel_size=3, padding=1), 
            nn.Tanh())

    def forward(self, input):
        downsampled = self.downsample(input)
        return self.upsample(downsampled)


class ColorizerV2(nn.Module):
    def __init__(self):
        super(ColorizerV2, self).__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(), 
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(), 

            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(), 
            
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 2, kernel_size=3, padding=1), 
            nn.Tanh()
        )

    def forward(self, input):
        downsampled = self.downsample(input)
        return self.upsample(downsampled)


class ColorizerV3(nn.Module):
    def __init__(self):
        super(ColorizerV3, self).__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(), 
            
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            
            nn.Conv2d(64, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(), 
            
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(), 

            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1), 
            nn.BatchNorm2d(8),
            nn.ReLU(), 
            
            nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2),
            nn.Conv2d(8, 2, kernel_size=3, padding=1), 
            nn.Tanh()
        )

    def forward(self, input):
        downsampled = self.downsample(input)
        return self.upsample(downsampled)


class ColorizerUNetV1(nn.Module):
    def __init__(self):
        super(ColorizerUNetV1, self).__init__()

        self.down1 = self.down(1, 32)
        self.down2 = self.down(32, 64)
        self.down3 = self.down(64, 128)
        self.down4 = self.down(128, 256)

        self.deconv1, self.conv1 = self.up(256, 128)
        self.dropout1 = nn.Dropout2d(0.2)
        self.deconv2, self.conv2 = self.up(128, 64)
        self.deconv3, self.conv3 = self.up(64, 32)

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )


    def down(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

    def up(self, in_dim, out_dim):
        return nn.ConvTranspose2d(in_dim, in_dim, kernel_size=2), nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

    def forward(self, input):
        down1 = self.down1(input)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        up1 = torch.cat((self.deconv1(down4), down3), dim=1)
        up1 = self.conv1(up1)
        up1 = self.dropout1(up1)

        up2 = torch.cat((self.deconv2(up1), down2), dim=1)
        up2 = self.conv2(up2)

        up3 = torch.cat((self.deconv3(up2), down1), dim=1)
        up3 = self.conv3(up3)
        
        output = self.conv4(up3)

        return output



class ColorizerUNetV2(nn.Module):
    def __init__(self):
        super(ColorizerUNetV2, self).__init__()

        self.down1 = self.down(1, 16)
        self.down2 = self.down(16, 32)
        self.down3 = self.down(32, 64)

        self.deconv1, self.conv1 = self.up(64, 32)
        self.dropout1 = nn.Dropout2d(0.2)
        self.deconv2, self.conv2 = self.up(32, 16)

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )


    def down(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

    def up(self, in_dim, out_dim):
        return nn.ConvTranspose2d(in_dim, in_dim, kernel_size=2), nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

    def forward(self, input):
        down1 = self.down1(input)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        up1 = torch.cat((self.deconv1(down3), down2), dim=1)
        up1 = self.conv1(up1)
        up1 = self.dropout1(up1)

        up2 = torch.cat((self.deconv2(up1), down1), dim=1)
        up2 = self.conv2(up2)
        
        output = self.conv3(up2)

        return output


models = {
    "v1": ColorizerV1,
    "v2": ColorizerV2,
    "v3": ColorizerV3,
    "unet-v1": ColorizerUNetV1,
    "unet-v2": ColorizerUNetV2
}
