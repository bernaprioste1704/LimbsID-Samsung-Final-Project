import os
import cv2
import time
import math as m
import time
#import mediapipe as mp
import torch
import torch.nn as nn
import torchvision.transforms as transforms

#from main import mp_drawing
from resizeImage import transform_data


transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor (3, 224, 224)
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize RGB
])


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


class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        # Using residual blocks for feature extraction
        self.conv = nn.Sequential(
            ResidualBlock(3, 32),  # Initial Conv + Residual Block
            nn.MaxPool2d(2, 2),  # 112x112

            ResidualBlock(32, 64),  # Residual Block
            nn.MaxPool2d(2, 2),  # 56x56

            ResidualBlock(64, 128),  # Residual Block
            nn.MaxPool2d(2, 2),  # 28x28

            ResidualBlock(128, 256),  # Residual Block
            nn.MaxPool2d(2, 2),  # 28x28

            ResidualBlock(256, 512),  # Residual Block
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

def predict(image,model,file):
    model.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        img_tensor = transform(image).unsqueeze(0)
        output = model(img_tensor)*224

    keypoints = output.squeeze()  # Remove extra dimension if necessary
    points = [(keypoints[i].item(), keypoints[i+1].item()) for i in range(0, len(keypoints), 2)]
    return points




def findDistance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    return int(180 / m.pi * theta)

def sendWarning():
    print("Warning: Poor posture detected!")


# Initialize frame counters
good_frames = 0
bad_frames = 0

# Font type and Colors
font = cv2.FONT_HERSHEY_SIMPLEX
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
yellow = (0, 255, 255)


connections = [
    (0, 1), (1, 2), (2, 6), (6, 3), (3, 4), (4, 5),  # legs
    (6, 7), (7, 8), (8, 9),  # body
    (7, 12), (12, 11), (11, 10),  # right arm
    (7, 13), (13, 14), (14, 15)  # left arm
]

model = PoseNet()
file = "posev0.01126.pth"
mp_drawing = mp.solutions.drawing_utils  # Utility for drawing

# Load video
target_video = 'input.mp4'
cap = cv2.VideoCapture(0) #2 streaming
fps = int(cap.get(cv2.CAP_PROP_FPS))
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_output = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

start_time = time.time()

while cap.isOpened():
    success, image_original = cap.read()
    if not success:
        break

    keypoints = None
    try:
        image, _, reverse1 = transform_data(image_original, [])
        keypoints = predict(image, model, file)  # OUR MODEL #
    except Exception as e:
        print(f"Error processing row: {e}")

    if not keypoints:
        sendWarning()
    else:


        #Laser pointer
        #x, y = reverse1(keypoints[7][0], keypoints[7][1])
        #cv2.circle(image_original, (x, y), 15,red, -1)  # Draw keypoint

        # Draw keypoints on image
        for keypoint in keypoints:
            x, y = reverse1(keypoint[0],keypoint[1])
            cv2.circle(image_original, (x, y), 5,red, -1)  # Draw keypoint


        for connection in connections:
            x1, y1 = reverse1(keypoints[connection[0]][0],keypoints[connection[0]][1])
            x2, y2 = reverse1(keypoints[connection[1]][0],keypoints[connection[1]][1])
            cv2.line(image_original, (x1, y1), (x2, y2), green, 2)
        #raw skeleton connections
        mp_drawing.draw_landmarks(image, lm, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=green, thickness=2, circle_radius=3),
                                  mp_drawing.DrawingSpec(color=yellow, thickness=2))

        # Get important body keypoints
        #l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        #l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        #r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        #r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        #l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        #l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
        #l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        #l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

        # Posture analysis
        #offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
        #if offset < 100:
        #    cv2.putText(image, 'Aligned', (w - 150, 30), font, 0.9, green, 2)
        #else:
        #    cv2.putText(image, 'Not Aligned', (w - 150, 30), font, 0.9, red, 2)

        #neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        #torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        #angle_text = f'Neck: {neck_inclination}  Torso: {torso_inclination}'
        #if neck_inclination < 40 and torso_inclination < 10:
        #    bad_frames = 0
        #    good_frames += 1
        #    cv2.putText(image, angle_text, (10, 30), font, 0.9, green, 2)
        #else:
        #    good_frames = 0
        #    bad_frames += 1
        #    cv2.putText(image, angle_text, (10, 30), font, 0.9, red, 2)
        #    sendWarning()

        #good_time = (1 / fps) * good_frames
        #bad_time = (1 / fps) * bad_frames
        #time_string = f'Good Posture Time: {round(good_time, 1)}s' if good_time > 0 else f'Bad Posture Time: {round(bad_time, 1)}s'
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        start_time = end_time

        cv2.putText(image_original, f"FPS: {int(fps)}", (10, h - 20), font, 2, red, 3)

    #video_output.write(image_original)
    cv2.imshow('Posture Detection', image_original)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_output.release()
cv2.destroyAllWindows()
