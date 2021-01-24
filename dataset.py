""" Dataset module
"""
import cv2
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class MaskDataset(Dataset):
    """ Masked faces dataset
        0 = 'no mask'
        1 = 'mask'
    """
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        
        self.transformations = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError('slicing is not supported')
        
        row = self.dataFrame.iloc[key]
        image = self.transformations(cv2.imread(row['image']))
        label = tensor([row['mask']], dtype=float)
        return image,label
    
    def __len__(self):
        return len(self.dataFrame.index)
