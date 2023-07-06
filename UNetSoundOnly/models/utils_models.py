import torch
import torchaudio.transforms as T
from torch import nn


class encode_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, double=False):
        super(encode_block, self).__init__()
        self.double = double
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )
        if self.double: # Quick bugfix without testing. DDP will not work if parameters are defined which are not used in the computation.
            self.conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, True)
                )

    def forward(self, x):
        x = self.down_conv(x)
        if self.double:
            x = self.conv(x)
        
        return x
    

class fc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(fc, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        x = self.fc(x)
        
        return x
    

class decode_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, double=False):
        super(decode_block, self).__init__()
        self.double = double
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding, stride = stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
            )
        
    def forward(self, x):
        x = self.up_conv(x)
        if self.double:
            x = self.conv(x)
        
        return x
