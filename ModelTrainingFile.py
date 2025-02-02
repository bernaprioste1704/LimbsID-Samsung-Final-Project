import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim

df = pd.read_pickle('dataset11231.pkl')
df = df.dropna().drop(['NAME','Scale','Activity','Category','ID'], axis=1)
#convert the columns upper neck_X upper neck_Y  head top_X  head top_Y to int64
df = df.astype({'upper neck_X': 'int64', 'upper neck_Y': 'int64', 'head top_X': 'int64', 'head top_Y': 'int64'})


#Construct the Dataset Object
keypoint_columns = [col for col in df.columns if col != 'image'] #todas as coluunas 
keypoints = df[keypoint_columns].values.astype(np.float32)
keypoints = keypoints / 224.0 #normaliza os pontos para 0-1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class PoseDataset(data.Dataset):
    def __init__(self, df):
        self.df = df
        self.images = df['image'].tolist()
        self.keypoints = keypoints
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_array = self.images[idx]
        img_tensor = transform(img_array) #converter para pythorch
        kp_tensor = torch.tensor(self.keypoints[idx])
        return img_tensor, kp_tensor
      
      
      
dataset = PoseDataset(df)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False)




class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
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


class ModelTraining:
    def __init__(self, model=PoseNet(), epochs=10, lr=0.00125, weight_decay=2e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model if self.device.type == "cpu" else model.to("cuda")
        self.criterion = nn.SmoothL1Loss()  # Used for training
        self.MSEcriterion = nn.MSELoss()  # Only for logging
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            train_smooth_loss, train_mse_loss = 0.0, 0.0
            
            for images, keypoints in train_loader:
                if self.device.type == 'cuda':
                    images, keypoints = images.to("cuda"), keypoints.to("cuda")
                    
                self.optimizer.zero_grad()
                outputs = self.model(images)
                
                smooth_loss = self.criterion(outputs, keypoints)
                mse_loss = self.MSEcriterion(outputs, keypoints)  # Just for monitoring
                
                smooth_loss.backward()
                self.optimizer.step()
                
                train_smooth_loss += smooth_loss.item()
                train_mse_loss += mse_loss.item()
            
            avg_train_smooth_loss = train_smooth_loss / len(train_loader)
            avg_train_mse_loss = train_mse_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss (SmoothL1): {avg_train_smooth_loss:.4f}, Train Loss (MSE): {avg_train_mse_loss:.4f}")
            
            # Validation
            self.model.eval()
            val_smooth_loss, val_mse_loss = 0.0, 0.0
            with torch.no_grad():
                for images, keypoints in val_loader:
                  
                  if self.device.type == 'cuda':
                    images, keypoints = images.to("cuda"), keypoints.to("cuda")
                  
                    outputs = self.model(images)
                    
                    smooth_loss = self.criterion(outputs, keypoints)
                    mse_loss = self.MSEcriterion(outputs, keypoints)
                    
                    val_smooth_loss += smooth_loss.item()
                    val_mse_loss += mse_loss.item()
            
            avg_val_smooth_loss = val_smooth_loss / len(val_loader)
            avg_val_mse_loss = val_mse_loss / len(val_loader)
            print(f"Val Loss (SmoothL1): {avg_val_smooth_loss:.4f}, Val Loss (MSE): {avg_val_mse_loss:.4f}")
