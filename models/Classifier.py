import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

from Model import ResidualBlock, ResNet

class Classifier: 
    
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
    
    def transform(self, img):
        trfm = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(32),
                transforms.ToTensor()])
        return trfm(img)
        
    def load_model(self):
        self.model.load_state_dict(torch.load('./resnet.ckpt'))
        self.model.eval()
    
    def predict(self, img):
        with torch.no_grad():
            prediction = self.model(img)
            predicted_class = np.argmax(prediction)
            print(f"Predicted class: {predicted_class}")
