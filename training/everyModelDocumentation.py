#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn


# First Model

# In[ ]:


class PoseNetV1(nn.Module):
    def __init__(self):
        super(PoseNetV1, self).__init__()
        self.conv = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112x112
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28
            
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 32) 
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# Here, we added one more layer and we added Dropout to the fully connected layer. We also added a Flatten layer to flatten the output of the convolutional layers before passing it to the fully connected layers.

# In[ ]:


class PoseNetV2(nn.Module):
    def __init__(self):
        super(PoseNetV2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112x112
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14
            
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 32) 
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# In[ ]:


class PoseNetV3(nn.Module):
    def __init__(self):
        super(PoseNetV3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112x112
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 32) 
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# We added batch normalization in each layer, Adaptive Pooling and a Tahn function at the end of the fully conected layers

# In[ ]:


class PoseNetV4(nn.Module):
    def __init__(self):
        super(PoseNetV4, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56x56

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))  # Adaptive pooling to make output size consistent
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout to prevent overfitting
            nn.Linear(512, 32),
            nn.Tanh()  # Normalizing keypoint predictions
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# 4 Layers -> 5 Layers
# 
# Tahn() -> Sigmoid()

# In[ ]:


class PoseNetV5(nn.Module):
    def __init__(self):
        super(PoseNetV5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56x56

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))  # Adaptive pooling to make output size consistent
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.50),  # Increased dropout to prevent overfitting
            nn.Linear(512, 32),
            nn.Sigmoid()  # Normalizing keypoint predictions
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# In[ ]:


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (identity mapping)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Adding the residual connection
        out = self.relu(out)
        return out

class ResPoseNet(nn.Module):
    def __init__(self):
        super(ResPoseNet, self).__init__()
        # Using residual blocks for feature extraction
        self.conv = nn.Sequential(
            ResidualBlock(3, 32),    # Initial Conv + Residual Block
            nn.MaxPool2d(2, 2),      # 112x112
            
            ResidualBlock(32, 64),   # Residual Block
            nn.MaxPool2d(2, 2),      # 56x56

            ResidualBlock(64, 128),  # Residual Block
            nn.MaxPool2d(2, 2),      # 28x28

            ResidualBlock(128, 256),  # Residual Block
            nn.MaxPool2d(2, 2),      # 28x28
            
            ResidualBlock(256, 512), # Residual Block
            nn.AdaptiveAvgPool2d((7, 7))  # 14x14 output
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.40),

            nn.Linear(1024, 32),  # Assuming 16 keypoints, each with x, y = 32 values
            nn.Sigmoid()  # Output keypoint coordinates between [0,1]
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

