import cv2
import numpy 
import torch
from pathlib import Path
from torch.utils.data import Dataset


class YOLODataset(Dataset):
    def __init__(self, img_dir: str, label_dir: str, img_size: int = 648, augment: bool = False):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size 
        self.augment = augment
        
        self.img_paths = sorted(list(self.img_dir.glob("*.jpg")))
        self.img_paths = [
            p for p in self.img_paths 
            if (self.label_dir / p.stem).with_suffix(".txt").exists()
        ]
        
    def __len__(self):
        return len(self.imp_paths)
    
    def __getitem__(self,idx):
        img_path = self.img_paths[idx]
        label_path = (self.label_dir / img_path.stem).with_suffix(".txt")
        
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(self.img_size,self.img_size))
        
        if self.augment:
            img = self._augment(img)
        img = img.astype(np.float32)/255.0
        img = np.transpose(img,(2,0,1))
        
        if label_path.exists():
            labels = np.loadtxt(label_path,dtype = np.float32).reshape(-1,5)
        else:
            labels = np.empty((0,5),dtype=np.float32)
            
        batch_idx = np.zeros((len(labels),1))
        targets = torch.from_numpy(np.hstack((batch_idx,labels)))
        
        return torch.from_numpy(img),targets,img_path
            
    def _augment(self,img):
        if np.random.random()>0.5:
            hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
            h,s,v = cv2.split(hsv)
            
            h = np.mod(h + np.random.randint(-10,10),180).astype(np.uint8)
            s  = np.clip(s+np.random.randint(-30,30),0,255).astype(np.uint8)
            v = np.clip (v +np.random.randint(-30,30),0,255).astype(np.uint8)
            
            img = cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2RGB)
        return img 