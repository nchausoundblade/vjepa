import sys
import os

# 1. CRITICAL: Force Python to see the V-JEPA root directory so 'src' works
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
import av  
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
        self.df = pd.read_csv(csv_path, sep=r'\s+|,', names=['path', 'label'], engine='python')
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['label']
        
        try:
            container = av.open(video_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not open video: {video_path}. Error: {e}")
            
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format='rgb24')
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            frames.append(img_tensor)
            
        container.close()
        
        if len(frames) == 0:
            raise ValueError(f"Video contains 0 decoded frames: {video_path}")
            
        video = torch.stack(frames)
        
        total_frames = video.size(0)
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        video = video[indices]

        if self.transform:
            transformed_frames = [self.transform(frame.float() / 255.0) for frame in video]
            video = torch.stack(transformed_frames)
            
        video = video.permute(1, 0, 2, 3)
        
        return video, label, video_path

# -------------------------------------------------------------------------
# 2. CONFIGURATION & CORE PIPELINE
# -------------------------------------------------------------------------
def main():
    CSV_PATH = "vjepa_dataset.csv" 
    
    # --- CHANGE THIS VARIABLE TO SWITCH MODELS ---
    # Options: 'ViT-L', 'ViT-H', 'ViT-H-384'
    MODEL_TYPE = 'ViT-H-384' 

    # Centralized configuration mapping for architecture, resolution, and weights
    MODEL_CONFIGS = {
        'ViT-L': {
            'arch': 'vit_large',
            'img_size': 224,
            'patch_size': 16,
            'checkpoint': "/Users/noahchau/Desktop/vitl16.pth.tar" # Ensure this path is correct
        },
        'ViT-H': {
            'arch': 'vit_huge',
            'img_size': 224,
            'patch_size': 16,
            'checkpoint': "/Users/noahchau/Desktop/vith16.pth.tar"
        },
        'ViT-H-384': {
            'arch': 'vit_huge',
            'img_size': 384,
            'patch_size': 16, # Assuming patch size 16 remains constant
            'checkpoint': "/Users/noahchau/Desktop/vith16-384.pth.tar" # Ensure this path is correct
        }
    }

    if MODEL_TYPE not in MODEL_CONFIGS:
        raise ValueError(f"Invalid MODEL_TYPE. Please choose from {list(MODEL_CONFIGS.keys())}")

    # Extract chosen configurations
    config = MODEL_CONFIGS[MODEL_TYPE]
    IMG_SIZE = config['img_size']
    CHECKPOINT_PATH = config['checkpoint']

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    print(f"Selected Model: {MODEL_TYPE} | Resolution: {IMG_SIZE}x{IMG_SIZE}")

    # Transforms now dynamically use the correct IMG_SIZE
    transform = Compose([
        Resize(IMG_SIZE),
        CenterCrop(IMG_SIZE),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = UltrasoundVideoDataset(csv_path=CSV_PATH, transform=transform, num_frames=16)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # -------------------------------------------------------------------------
    # 3. INITIALIZE BACKBONE AND LOAD WEIGHTS
    # -------------------------------------------------------------------------
    print(f"Initializing {config['arch']} backbone...")
    
    # Dynamically select the correct architecture string from the dict registry
    model = vit.__dict__[config['arch']](
        img_size=IMG_SIZE,
        patch_size=config['patch_size'],
        num_frames=16,
        tubelet_size=2
    )

    print(f"Loading weights from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    if 'target_encoder' in checkpoint:
        state_dict = checkpoint['target_encoder']
    else:
        state_dict = checkpoint

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint loaded status: {msg}")
    
    model = model.to(DEVICE)
    model.eval() 

    # -------------------------------------------------------------------------
    # 4. LOOP & FEATURE EXTRACTION
    # -------------------------------------------------------------------------
    all_embeddings = []
    all_labels = []
    all_paths = []

    total_videos = len(dataloader)
    print(f"🚀 Starting extraction loop for {total_videos} ultrasound sweeps...\n")
    
    with torch.no_grad():
        # Added enumerate to track the current index (starting at 1)
        for idx, (videos, labels, paths) in enumerate(dataloader, start=1):
            current_path = paths[0]
            video_filename = os.path.basename(current_path)
            
            # Print current progress cleanly without making a massive wall of text
            print(f"[{idx}/{total_videos}] Extracting from: {video_filename} ... ", end="", flush=True)
            
            videos = videos.to(DEVICE)
            
            # Forward pass through the transformer backbone
            features = model(videos) 
            
            # Global Average Pooling over the tokens/patches to reduce down to a single 1D vector per video
            if len(features.shape) == 3: # (Batch, Tokens, Dim)
                features = torch.mean(features, dim=1) # Collapse space/time tokens -> (Batch, Dim)
                
            # Flatten to 1D and move back to system memory
            embedding_vector = features.squeeze(0).cpu().numpy()
            
            all_embeddings.append(embedding_vector)
            all_labels.append(labels.item())
            all_paths.append(current_path)
            
            # Finalizes the line once the video is finished processing
            print("Done! ✅")

    # Convert lists into pure numpy arrays for visualization tools
    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)

    print(f"\n🎉 Extraction Complete! Generated shape matrix: {all_embeddings.shape}")

    # -------------------------------------------------------------------------
    # 5. SAVE MATRIX DATA DIRECTLY TO DESKTOP
    # -------------------------------------------------------------------------
    # Appending the model type to the output file names so you don't overwrite your work
    np.save(f"/Users/noahchau/Desktop/ultrasound_embeddings_{MODEL_TYPE}.npy", all_embeddings)
    np.save(f"/Users/noahchau/Desktop/ultrasound_labels_{MODEL_TYPE}.npy", all_labels)
    print(f"Saved 'ultrasound_embeddings_{MODEL_TYPE}.npy' and labels to Desktop. Ready for t-SNE/UMAP!")

if __name__ == "__main__":
    main()