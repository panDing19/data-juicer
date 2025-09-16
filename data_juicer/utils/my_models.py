import os
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """MLP model for aesthetic score prediction."""
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticScorer(nn.Module):
    """Model to predict aesthetic scores for images and videos."""
    def __init__(self, input_size, device, model_path=None):
        super().__init__()
        self.mlp = MLP(input_size)
        self.clip, self.preprocess = clip.load("ViT-L/14", device=device)
        
        # Load pre-trained aesthetic model weights
        if model_path is not None:
            if os.path.exists(model_path):
                self.mlp.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded aesthetic model weights from {model_path}")
            else:
                print(f"Warning: Aesthetic model weights not found at {model_path}. Using uninitialized weights.")
        else:
            # Try default path as fallback
            default_path = os.path.expanduser("~/Code/ray_for_opensora/pretrained_models/aesthetic.pth")
            if os.path.exists(default_path):
                self.mlp.load_state_dict(torch.load(default_path, map_location=device))
                print(f"Loaded aesthetic model weights from default path: {default_path}")
            else:
                print(f"Warning: No model path specified and default path not found. Using uninitialized weights.")
        
        self.eval()
        self.to(device)

    def forward(self, x):
        image_features = self.clip.encode_image(x)
        image_features = F.normalize(image_features, p=2, dim=-1).float()
        return self.mlp(image_features)