import os
import numpy as np
import pandas as pd
import umap
import plotly.express as px
from sklearn.manifold import trustworthiness

# =========================================================================
# CONFIGURATION SETTINGS
# =========================================================================
# Which metadata category do you want to colour the dots by?
# Options: 'Diagnosis', 'Study', 'Sweep_Number', 'US_Model', 'US_Generation', 
#          'Photometric_Mode', 'Frame_Range', 'FPS_Range', 'Angle'
COLOR_BY = 'Photometric_Mode'  

# UMAP Hyperparameters
n_neighbors_val = 5
min_dist_val = 0.1
n_components_val = 2
metric_val = 'manhattan'  # 'euclidean', 'manhattan', 'cosine', etc.
seed_val = 10


def main():

    # --- CHANGE THIS TO MATCH THE EMBEDDINGS YOU WANT TO LOAD ---
    MODEL_TYPE = 'ViT-H-384'  # Options: 'ViT-L', 'ViT-H', 'ViT-H-384'

    # 1. Load the generated matrices dynamically
    try:
        embeddings = np.load(f"/Users/noahchau/Desktop/ultrasound_embeddings_{MODEL_TYPE}.npy")
        labels = np.load(f"/Users/noahchau/Desktop/ultrasound_labels_{MODEL_TYPE}.npy")

        n_samples, feature_dim = embeddings.shape
        print(f"Loaded {n_samples} embeddings with feature dimension size: {feature_dim}")
        
        # Automated Native Architecture Identification Profile Engine
        if feature_dim == 1024:
            ARCHITECTURE = "V-JEPA (ViT-L)"
        elif feature_dim == 1280:
            ARCHITECTURE = "V-JEPA (ViT-H)"
        elif feature_dim == 768:
            ARCHITECTURE = "V-JEPA (ViT-B)"
        else:
            ARCHITECTURE = f"Custom-Net (Dim-{feature_dim})"
            
        print(f"🤖 Auto-detected Backbone Profile: {ARCHITECTURE}")
        
    except FileNotFoundError:
        print(f"Error: Could not find .npy files for {MODEL_TYPE}. Run 'extract_embeddings.py' first!")
        return


    # 2. Extract filenames from your original CSV for the hover tooltips
    try:
        # Reads the CSV just like your dataset class did
        df_csv = pd.read_csv("vjepa_dataset.csv", sep=r'\s+|,', names=['path', 'label'], engine='python')
        filenames = [os.path.basename(p) for p in df_csv['path']]
    except FileNotFoundError:
        print("Error: Could not find 'vjepa_dataset.csv'. Ensure it is in the same folder.")
        return

    # Map the 0 and 1 integers to actual clinical names for the legend
    # (Adjust these strings if your labels mean something different!)
    diagnoses = ["Abnormal" if label == 1 else "Normal" for label in labels]

    print(f"Loaded {embeddings.shape[0]} embeddings. Computing UMAP projection...")

    # Initialize metadata containers for all 8 components
    studies, sweep_numbers, us_models, us_generations = [], [], [], []
    photometrics, frame_ranges, fps_ranges, angles = [], [], [], []


    # 3. Parse the strict 8-part naming layout
    for name in filenames:
        clean_name = os.path.splitext(name)[0]
        parts = clean_name.split('_')
        
        # Expecting exactly 8 parts after splitting by underscores
        if len(parts) >= 8:
            studies.append(parts[0])         # Part 1: Study ID
            sweep_numbers.append(parts[1])   # Part 2: Cine Sweep ID
            us_models.append(parts[2])       # Part 3: Ultrasound Brand/Model
            us_generations.append(parts[3])  # Part 4: Machine Generation
            photometrics.append(parts[4])    # Part 5: Photometric Imaging Mode
            angles.append(parts[7])          # Part 8: Scan Orientation Angle
            
            # Part 6: Parse Frames (Strips the trailing 'f' to evaluate values)
            try:
                raw_frames = parts[5].lower().rstrip('f')
                frame_count = int(raw_frames)
                if frame_count <= 50:
                    frame_ranges.append("0-50 Frames")
                elif frame_count <= 150:
                    frame_ranges.append("51-150 Frames")
                else:
                    frame_ranges.append("151+ Frames")
            except ValueError:
                frame_ranges.append("Unknown Frames")

            # Part 7: Parse FPS (Strips the trailing 'fps' to evaluate ranges)
            try:
                raw_fps = parts[6].lower().replace('fps', '')
                fps = float(raw_fps)
                if fps <= 15:
                    fps_ranges.append("0-15 FPS")
                elif fps <= 30:
                    fps_ranges.append("16-30 FPS")
                elif fps <= 60:
                    fps_ranges.append("31-60 FPS")
                else:
                    fps_ranges.append("61+ FPS")
            except ValueError:
                fps_ranges.append("Unknown FPS")
                
        else:
            # Fallback safety array padding for non-conformant naming strings
            studies.append("Malformed_Name")
            sweep_numbers.append("Unknown")
            us_models.append("Unknown")
            us_generations.append("Unknown")
            photometrics.append("Unknown")
            frame_ranges.append("Unknown")
            fps_ranges.append("Unknown")
            angles.append("Unknown")

    print(f"Computing UMAP projection (n_neighbors={n_neighbors_val}, min_dist={min_dist_val})...")

    # 4. Configure UMAP 
    # CHANGE VARIABLES AT TOP TO EXPERIMENT WITH DIFFERENT SETTINGS 

    reducer = umap.UMAP(
        n_neighbors=n_neighbors_val,       
        min_dist=min_dist_val,        
        n_components=n_components_val,      
        metric=metric_val,     
        random_state=seed_val      
    )

    embeddings_2d = reducer.fit_transform(embeddings)

    # Calculate how well the 2D map trusted the original high-dimensional space
    trust_score = trustworthiness(embeddings, embeddings_2d, n_neighbors=n_neighbors_val)

    # 5. Build full metadata tracking DataFrame
    plot_df = pd.DataFrame({
        'UMAP X': embeddings_2d[:, 0],
        'UMAP Y': embeddings_2d[:, 1],
        'Filename': filenames,
        'Diagnosis': diagnoses,
        'Study': studies,
        'Sweep_Number': sweep_numbers,
        'US_Model': us_models,
        'US_Generation': us_generations,
        'Photometric_Mode': photometrics,
        'Frame_Range': frame_ranges,
        'FPS_Range': fps_ranges,
        'Angle': angles
    })

    # 6. Generate Plotly Map with custom legend sorting and colours
    umap_settings_title = (
        f"V-JEPA Latent Space Map (UMAP : 8-Part Metadata Specification)<br>"
        f"<sub><b>Colored By:</b> {COLOR_BY} | <b>n_neighbors:</b> {n_neighbors_val} | "
        f"<b>min_dist:</b> {min_dist_val} | <b>metric:</b> {metric_val} | <b>seed:</b> {seed_val}</sub>"
        f"<br><sub><b>Approximate Information Loss:</b> {(1.0 - trust_score) * 100:.1f}%</sub>"
    )

    # Initialize empty controls for general categories
    custom_orders = None
    custom_color_map = None

    # Apply specialized grouping logic ONLY when analyzing the 'Angle' category
    if COLOR_BY == 'Angle':
        # A. Force the Legend Sorting Order
        custom_orders = {
            'Angle': ['RT-TRV', 'LT-TRV', 'ML-TRV', 'RT-SAG', 'LT-SAG', 'TBD', 'Unknown']
        }

    # B. Map Cohesive, High-Contrast Color Palettes
        custom_color_map = {
            'RT-TRV': '#002f6c',       # Deep Navy Blue
            'LT-TRV': '#1f77b4',       # Standard Bright Blue
            'ML-TRV': '#00efff',       # High-Contrast Electric Cyan
            'RT-SAG': '#ff4500',       # Vibrant Orange-Red
            'LT-SAG': '#ffaa00',       # Bright Gold-Orange
            'SAG': '#ff7f0e',          # Medium Orange (For any generic sagittal scans without clear RT/LT)
            'TBD': '#7f7f7f',          # Muted Slate Gray (For your un-evaluated files)
            'Unknown': '#bcbd22'       # Olive Green
        }

    elif COLOR_BY == 'Study':
        custom_orders = {
            'Study': ['study0', 'study1', 'study2', 'study3', 'study4', 'study5', 'study6', 'study7', 'study8', 'study9', 'study10']
        }
        custom_color_map = {
            'study0': "#4e0808",
            'study1': "#b41f1f",
            'study2': '#ff7f0e', 
            'study3': "#d9dd22",
            'study4': "#27d644", 
            'study5': "#1B6738", 
            'study5': "#1fbdba",
            'study6': "#1a75a7", 
            'study7': "#4f36de",
            'study8': "#a830df",
            'study9': "#a11d87",
            'study10': "#e6759a",
        }



    # If you want to customize other views later (like machines), you can add another 'elif' block here:
    elif COLOR_BY == 'Diagnosis':
        custom_color_map = {'Normal': '#2ca02c', 'Abnormal': '#d62728'} # Green vs Red

    fig = px.scatter(
        plot_df,
        x='UMAP X',
        y='UMAP Y',
        color=COLOR_BY,        
        hover_name='Filename',
        hover_data=[
            'Diagnosis', 'Study', 'Sweep_Number', 'US_Model', 
            'US_Generation', 'Photometric_Mode', 'Frame_Range', 'FPS_Range', 'Angle'
        ], 
        title=umap_settings_title,
        category_orders=custom_orders,         # <-- Injects the custom sorting list
        color_discrete_map=custom_color_map,   # <-- Injects the specific color mappings
        # Fallback ultra-high contrast palette if you color by something else (like Study)
        color_discrete_sequence=px.colors.qualitative.Dark24 
    )

    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        template="plotly_white",
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial")
    )

    # 7. Optimized Output Settings with Dynamic Metadata Filenames
    # Construct a clean, structured filename for the downloaded image
    download_filename = (
        f"umap_{COLOR_BY}_"
        f"neighbors{n_neighbors_val}_"
        f"mindist{min_dist_val}_"
        f"{metric_val}_"
        f"seed{seed_val}_"
        f"InfoLoss{(1.0 - trust_score) * 100:.1f}%"
    )

    export_config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': download_filename, # Applies the dynamic string
            'height': 1080,            
            'width': 1920,             
            'scale': 3                 # Keeps the 3x crisp sharpness multiplier
        }
    }

    # Pass the config block directly into the show function
    fig.show(config=export_config)
    
    # Keeps your web export active
    fig.write_html("interactive_umap_map.html", include_plotlyjs='cdn')
    print(f"Successfully generated map! Image download configured as: '{download_filename}.png'")
if __name__ == "__main__":
    main()