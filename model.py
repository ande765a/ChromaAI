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


models = {
    "v1": ColorizerV1,
    "v2": ColorizerV2,
    "v3": ColorizerV3   
}
