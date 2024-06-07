import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

class ClassificationLoss(nn.Module):
    def __init__(self, classes, target_cols, device):
        super().__init__()
        self.len = len(classes)
        self.target_cols = target_cols
        self.device = device
        
        self.c_matrix = self.make_class_matrix(classes)

    def forward(self, y_pred_batch, y_batch):
        y_pred_batch = F.softmax(y_pred_batch, dim=1)
        class_distances = self.c_matrix[y_batch]
        losses = torch.sum(y_pred_batch * class_distances, dim=1)
        return losses.mean()
    
    def d(self,x,y):
        # np.mean changed to np.sum to reflect Euclidean distance more closely
        return (np.sum((x - y) ** 2, axis=2))
    
    def make_class_matrix(self, classes):
        class_features = np.array(classes[self.target_cols])

        distances = self.d(class_features[:, None], class_features)
        print(np.max(distances))
        print(np.stack(distances).shape)
                        
        return torch.from_numpy(np.stack(distances)).to(self.device)