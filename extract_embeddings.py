import sys
import os

# 1. CRITICAL: Force Python to see the V-JEPA root directory so 'src' works
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
import av  # Modern alternative to the deprecated torchvision.io.read_video
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

# Import Meta's vision transformer architecture 
import src.models.vision_transformer as vit

# -------------------------------------------------------------------------
# 1. SIMPLE VIDEO DATALOADER FOR ULTRASOUND CINE SWEEPS
# -------------------------------------------------------------------------
class UltrasoundVideoDataset(Dataset):
    """
    An updated dataset class that reads ultrasound sweeps using PyAV,
    bypassing the deprecated and removed torchvision.io.read_video function.
    """
    def __init__(self, csv_path, transform=None, num_frames=16):
        # Reads standard space-separated or comma-separated CSV: [path label]
        self.df = pd.read_csv(csv_path, sep=r'\s+|,', names=['path', 'label'], engine='python')
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['label']
        
        # Open video container with PyAV
        try:
            container = av.open(video_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not open video: {video_path}. Error: {e}")
            
        frames = []
        # Extract all video frames as standard numpy arrays
        for frame in container.decode(video=0):
            # Convert PyAV frame to an RGB numpy array
            img = frame.to_ndarray(format='rgb24')
            # Convert to PyTorch tensor format: (Channels, Height, Width)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            frames.append(img_tensor)
            
        container.close()
        
        if len(frames) == 0:
            raise ValueError(f"Video contains 0 decoded frames: {video_path}")
            
        # Convert list of tensors into a single video tensor: (Total_Frames, C, H, W)
        video = torch.stack(frames)
        
        # Downsample or pad video uniformly along the time dimension to get exactly `num_frames`
        total_frames = video.size(0)
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        video = video[indices] # Shape: (num_frames, C, H, W)

        # Apply spatial transformations frame by frame
        if self.transform:
            # Scale uint8 values to float [0, 1] matching V-JEPA requirements
            transformed_frames = [self.transform(frame.float() / 255.0) for frame in video]
            video = torch.stack(transformed_frames) # Shape: (T, C, H, W)
            
        # V-JEPA expects video tensors shaped as (C, T, H, W) internally 
        video = video.permute(1, 0, 2, 3)
        
        return video, label, video_path

# -------------------------------------------------------------------------
# 2. CONFIGURATION & CORE PIPELINE
# -------------------------------------------------------------------------
def main():
    # Set your paths here
    CSV_PATH = "vjepa_dataset.csv"  # Path to your list of 41 videos
    CHECKPOINT_PATH = "/Users/noahchau/Desktop/vitl16.pth.tar" # Path to checkpoint
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")

    # Standard ImageNet / VideoMix transform setup matching V-JEPA defaults
    transform = Compose([
        Resize(224),
        CenterCrop(224),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize data loader
    dataset = UltrasoundVideoDataset(csv_path=CSV_PATH, transform=transform, num_frames=16)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # Batch size 1 to track video paths cleanly

    # -------------------------------------------------------------------------
    # 3. INITIALIZE BACKBONE AND LOAD WEIGHTS
    # -------------------------------------------------------------------------
    print("Initializing ViT-Large backbone...")
    # Initialize the architecture using Meta's dictionary registry lookup
    model = vit.__dict__['vit_large'](
        img_size=224,
        patch_size=16,
        num_frames=16,
        tubelet_size=2
    )

    print(f"Loading weights from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # Target encoder weights are wrapped in a dict key usually named 'target_encoder'
    if 'target_encoder' in checkpoint:
        state_dict = checkpoint['target_encoder']
    else:
        state_dict = checkpoint

    # Clean up standard DDP or framework prefixes if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}

    # Load into the model topology
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint loaded status: {msg}")
    
    model = model.to(DEVICE)
    model.eval() # Force Evaluation Mode

    # -------------------------------------------------------------------------
    # 4. LOOP & FEATURE EXTRACTION
    # -------------------------------------------------------------------------
    all_embeddings = []
    all_labels = []
    all_paths = []

    print(f"Extracting features from {len(dataset)} ultrasound sweeps...")
    
    with torch.no_grad():
        for videos, labels, paths in dataloader:
            videos = videos.to(DEVICE) # Shape: (1, 3, 16, 224, 224)
            
            # Forward pass through the transformer backbone
            features = model(videos) 
            
            # Global Average Pooling over the tokens/patches to reduce down to a single 1D vector per video
            if len(features.shape) == 3: # (Batch, Tokens, Dim)
                features = torch.mean(features, dim=1) # Collapse space/time tokens -> (Batch, Dim)
                
            # Flatten to 1D and move back to system memory
            embedding_vector = features.squeeze(0).cpu().numpy()
            
            all_embeddings.append(embedding_vector)
            all_labels.append(labels.item())
            all_paths.append(paths[0])

    # Convert lists into pure numpy arrays for visualization tools
    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)

    print(f"Extraction Complete! Generated shape matrix: {all_embeddings.shape}")

    # -------------------------------------------------------------------------
    # 5. SAVE MATRIX DATA DIRECTLY TO DESKTOP
    # -------------------------------------------------------------------------
    np.save("/Users/noahchau/Desktop/ultrasound_embeddings.npy", all_embeddings)
    np.save("/Users/noahchau/Desktop/ultrasound_labels.npy", all_labels)
    print("Saved 'ultrasound_embeddings.npy' and 'ultrasound_labels.npy' to Desktop. Ready for t-SNE/UMAP!")

if __name__ == "__main__":
    main()