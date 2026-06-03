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
    # Centralized Directory Mappings
    CHECKPOINT_DIR = "/Users/noahchau/Desktop/V-JEPA/Model-Checkpoints/"
    OUTPUT_DIR = "/Users/noahchau/Desktop/V-JEPA/Embeddings-and-Labels/"
    CSV_PATH = "/Users/noahchau/Desktop/V-JEPA/jepa/vjepa_dataset.csv" 

    # Ensure output directory exists before running
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Dictionary map connecting your file naming convention to structural model keys
    FILENAME_TO_MODEL = {
        "vith16-384.pth.tar": "ViT-H-384",
        "vith16.pth.tar": "ViT-H",
        "vitl16.pth.tar": "ViT-L"
    }
    
    found_checkpoints = {}

    # Scan the folder and parse exact matching keys
    if os.path.exists(CHECKPOINT_DIR):
        for filename in os.listdir(CHECKPOINT_DIR):
            if filename in FILENAME_TO_MODEL:
                model_key = FILENAME_TO_MODEL[filename]
                found_checkpoints[model_key] = os.path.join(CHECKPOINT_DIR, filename)
    
    VALID_MODELS = sorted(list(found_checkpoints.keys()))

    # Fallback structure logic just in case paths get disjointed/moved
    if not VALID_MODELS:
        print(f"⚠️ Warning: Checkpoint files not detected inside {CHECKPOINT_DIR}")
        print("Expected filenames: vitl16.pth.tar, vith16.pth.tar, vith16-384.pth.tar")
        print("Falling back to absolute default configurations.")
        VALID_MODELS = ['ViT-L', 'ViT-H', 'ViT-H-384']
        found_checkpoints = {
            'ViT-L': os.path.join(CHECKPOINT_DIR, "vitl16.pth.tar"),
            'ViT-H': os.path.join(CHECKPOINT_DIR, "vith16.pth.tar"),
            'ViT-H-384': os.path.join(CHECKPOINT_DIR, "vith16-384.pth.tar")
        }

    # Prompt user dynamically using only verified models
    print("Available model checkpoints found:", ", ".join(VALID_MODELS))
    while True:
        user_input = input(f"Enter MODEL_TYPE (default '{VALID_MODELS[0]}'): ").strip()
        if not user_input:
            MODEL_TYPE = VALID_MODELS[0]
            break
        elif user_input in VALID_MODELS:
            MODEL_TYPE = user_input
            break
        print(f"❌ Invalid choice. Please choose from available checkpoints: {VALID_MODELS}")

    # Centralized architecture rules mapped dynamically to your model selection
    MODEL_CONFIGS = {
        'ViT-L':      {'arch': 'vit_large', 'img_size': 224, 'patch_size': 16},
        'ViT-H':      {'arch': 'vit_huge',  'img_size': 224, 'patch_size': 16},
        'ViT-H-384':  {'arch': 'vit_huge',  'img_size': 384, 'patch_size': 16}
    }

    # Combine file discovery with the structural parameters
    config = MODEL_CONFIGS[MODEL_TYPE]
    IMG_SIZE = config['img_size']
    CHECKPOINT_PATH = found_checkpoints[MODEL_TYPE]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {DEVICE}")
    print(f"Selected Model: {MODEL_TYPE} | Resolution: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Target Checkpoint File: {os.path.basename(CHECKPOINT_PATH)}")

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

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        if 'target_encoder' in checkpoint:
            state_dict = checkpoint['target_encoder']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}

        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded status: {msg}")
    else:
        print(f"⚠️ Warning: Checkpoint file could not be verified at {CHECKPOINT_PATH}. Using random weights.")
    
    model = model.to(DEVICE)
    model.eval() 

    # -------------------------------------------------------------------------
    # 4. LOOP & FEATURE EXTRACTION
    # -------------------------------------------------------------------------
    all_embeddings = []
    all_labels = []
    all_paths = []

    total_videos = len(dataloader)
    print(f"\n🚀 Starting extraction loop for {total_videos} ultrasound sweeps...\n")
    
    with torch.no_grad():
        for idx, (videos, labels, paths) in enumerate(dataloader, start=1):
            current_path = paths[0]
            video_filename = os.path.basename(current_path)
            
            print(f"[{idx}/{total_videos}] Extracting from: {video_filename} ... ", end="", flush=True)
            
            videos = videos.to(DEVICE)
            
            # Forward pass through the transformer backbone
            features = model(videos) 
            
            # Global Average Pooling over the tokens/patches
            if len(features.shape) == 3: 
                features = torch.mean(features, dim=1) 
                
            embedding_vector = features.squeeze(0).cpu().numpy()
            
            all_embeddings.append(embedding_vector)
            all_labels.append(labels.item())
            all_paths.append(current_path)
            
            print("Done! ✅")

    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)

    print(f"\n🎉 Extraction Complete! Generated shape matrix: {all_embeddings.shape}")

    # -------------------------------------------------------------------------
    # 5. SAVE MATRIX DATA DIRECTLY TO SPECIFIED OUTPUT DIRECTORY
    # -------------------------------------------------------------------------
    save_emb_path = os.path.join(OUTPUT_DIR, f"ultrasound_embeddings_{MODEL_TYPE}.npy")
    save_lbl_path = os.path.join(OUTPUT_DIR, f"ultrasound_labels_{MODEL_TYPE}.npy")

    np.save(save_emb_path, all_embeddings)
    np.save(save_lbl_path, all_labels)
    
    print(f"\nSaved Matrix Data to Matrix Directory Location:\n -> {save_emb_path}\n -> {save_lbl_path}\nReady for UMAP!")

if __name__ == "__main__":
    main()