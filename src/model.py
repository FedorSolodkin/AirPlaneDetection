import torch 
import torch.nn as nn 
from ultralytics import YOLO 

class AirplaneDetector:
    
    def __init__ (self, model_name: str = "yolov8n",pretrained: bool = True):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if pretrained:
            self.model = YOLO(f"{model_name}.pt")
        else:
            self.model = YOLO()
        
        self.model.to(self.device)
        
    def get_torch_model(self):
        return self.model.model
    def train(self,**kwargs):
        return self.model.train(**kwargs)
    def val(self,**kwargs):
        return self.model.val(**kwargs)
    def predict(self,source,**kwargs):
        return self.model.predict(source,**kwargs)
    def save(self, path: str):
        """Save model weights"""
        torch.save(self.model.model.state_dict(), path)
    
    def load(self, path: str):
        """Load model weights"""
        self.model.model.load_state_dict(torch.load(path, map_location=self.device))