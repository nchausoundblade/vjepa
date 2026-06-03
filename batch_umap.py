import os
import sys
import numpy as np
import pandas as pd
import umap
import plotly.express as px
from sklearn.manifold import trustworthiness

def main():
    # --- AUTOMATIC ARCHITECTURE DISCOVERY & PROMPT ---
    CHECKPOINT_DIR = "/Users/noahchau/Desktop/V-JEPA/Model-Checkpoints/"
    EMBEDDINGS_DIR = "/Users/noahchau/Desktop/V-JEPA/Embeddings-and-Labels/"
    CSV_PATH = "/Users/noahchau/Desktop/V-JEPA/jepa/vjepa_dataset.csv"
    output_dir = "/Users/noahchau/Desktop/V-JEPA/umap_batch_results"

    # Map file strings exactly as they exist in your Model-Checkpoints folder
    FILENAME_TO_MODEL = {
        "vith16-384.pth.tar": "ViT-H-384",
        "vith16.pth.tar": "ViT-H",
        "vitl16.pth.tar": "ViT-L"
    }
    
    found_checkpoints = {}
    if os.path.exists(CHECKPOINT_DIR):
        for filename in os.listdir(CHECKPOINT_DIR):
            if filename in FILENAME_TO_MODEL:
                model_key = FILENAME_TO_MODEL[filename]
                found_checkpoints[model_key] = os.path.join(CHECKPOINT_DIR, filename)
    
    VALID_MODELS = sorted(list(found_checkpoints.keys()))

    # Dynamic fallback script guardrail
    if not VALID_MODELS:
        print(f"⚠️ Warning: Checkpoint files not detected inside {CHECKPOINT_DIR}")
        print("Defaulting validation parameters to standard specifications.")
        VALID_MODELS = ['ViT-L', 'ViT-H', 'ViT-H-384']

    print("Available V-JEPA models discovered:", ", ".join(VALID_MODELS))
    while True:
        user_input = input(f"Enter MODEL_TYPE to plot (default '{VALID_MODELS[0]}'): ").strip()
        if not user_input:
            MODEL_TYPE = VALID_MODELS[0]
            break
        elif user_input in VALID_MODELS:
            MODEL_TYPE = user_input
            break
        print(f"❌ Invalid choice. Choose from discovered matrix variants: {VALID_MODELS}")

    # Build exact paths for your matrix layers inside the Embeddings-and-Labels folder
    embedding_file_path = os.path.join(EMBEDDINGS_DIR, f"ultrasound_embeddings_{MODEL_TYPE}.npy")
    label_file_path = os.path.join(EMBEDDINGS_DIR, f"ultrasound_labels_{MODEL_TYPE}.npy")

    # 1. Load the generated matrices dynamically
    try:
        embeddings = np.load(embedding_file_path)
        labels = np.load(label_file_path)

        n_samples, feature_dim = embeddings.shape
        print(f"\nLoaded {n_samples} embeddings with feature dimension size: {feature_dim}")
        
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
        print(f"\n❌ Error: Could not find matrix records for {MODEL_TYPE}.")
        print(f"Looked for files:\n -> {embedding_file_path}\n -> {label_file_path}")
        print("Run your unified extraction script first!")
        return

    # 2. Extract filenames and parse metadata
    try:
        df_csv = pd.read_csv(CSV_PATH, sep=r'\s+|,', names=['path', 'label'], engine='python')
        filenames = [os.path.basename(p) for p in df_csv['path']]
    except FileNotFoundError:
        print(f"❌ Error: Could not find dataset records at {CSV_PATH}.")
        return

    diagnoses = ["Abnormal" if label == 1 else "Normal" for label in labels]

    # Initialize metadata containers for all 8 components
    studies, sweep_numbers, us_models, us_generations = [], [], [], []
    photometrics, frame_ranges, fps_ranges, angles = [], [], [], []

    # 3. Parse the strict naming layout, ignoring "cropped_" if present
    for name in filenames:
        clean_name = os.path.splitext(name)[0]
        
        # FIX: Check if the filename starts with 'cropped_' and strip it off to preserve index structural alignment
        if clean_name.startswith("cropped_"):
            clean_name = clean_name[len("cropped_"):]
            
        parts = clean_name.split('_')
        
        if len(parts) >= 8:
            studies.append(parts[0])         
            sweep_numbers.append(parts[1])   
            us_models.append(parts[2])       
            us_generations.append(parts[3])  
            photometrics.append(parts[4])    
            angles.append(parts[7])          
            
            try:
                raw_frames = parts[5].lower().rstrip('f')
                frame_count = int(raw_frames)
                if frame_count <= 50: frame_ranges.append("0-50 Frames")
                elif frame_count <= 150: frame_ranges.append("51-150 Frames")
                else: frame_ranges.append("151+ Frames")
            except ValueError:
                frame_ranges.append("Unknown Frames")

            try:
                raw_fps = parts[6].lower().replace('fps', '')
                fps = float(raw_fps)
                if fps <= 15: fps_ranges.append("0-15 FPS")
                elif fps <= 30: fps_ranges.append("16-30 FPS")
                elif fps <= 45: fps_ranges.append("31-45 FPS")
                elif fps <= 60: fps_ranges.append("46-60 FPS")
                else: fps_ranges.append("61+ FPS")
            except ValueError:
                fps_ranges.append("Unknown FPS")
        else:
            studies.append("Malformed_Name")
            sweep_numbers.append("Unknown")
            us_models.append("Unknown")
            us_generations.append("Unknown")
            photometrics.append("Unknown")
            frame_ranges.append("Unknown")
            fps_ranges.append("Unknown")
            angles.append("Unknown")

    base_df = pd.DataFrame({
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

    # =========================================================================
    # MULTI-LATENT SPACE EXPERIMENT MATRIX SPECIFICATION
    # =========================================================================
    experiments = [
        {
            "name": "Clinical_Ideal",
            "neighbors": 7, "min_dist": 0.25, "metric": "cosine", "seeds": [1, 100],
            "colors": ["Diagnosis", "Angle"]
        },
        {
            "name": "Local_Cluster",
            "neighbors": 3, "min_dist": 0.05, "metric": "cosine", "seeds": [80, 85],
            "colors": ["Study", "Angle"]
        },
        {
            "name": "Global_Topology",
            "neighbors": 12, "min_dist": 0.4, "metric": "cosine", "seeds": [25, 75],
            "colors": ["Study", "Angle"]
        },
        {
            "name": "Magnitude_Test",
            "neighbors": 6, "min_dist": 0.15, "metric": "euclidean", "seeds": [40, 60],
            "colors": ["FPS_Range", "Frame_Range"] 
        },
        {
            "name": "Pixel_Noise_Grid",
            "neighbors": 5, "min_dist": 0.1, "metric": "manhattan", "seeds": [10, 20],
            "colors": ["Photometric_Mode", "Angle"]
        },
        {
            "name": "Device_Verification",
            "neighbors": 6, "min_dist": 0.3, "metric": "cosine", "seeds": [42, 67],
            "colors": ["US_Model", "Angle"]
        }
    ]

    os.makedirs(output_dir, exist_ok=True)

    total_runs = sum(len(exp["seeds"]) * len(exp["colors"]) for exp in experiments)
    current_run = 0

    print(f"🚀 Initializing automated pipeline. Computing {total_runs} custom UMAP variations...\n")

    for exp in experiments:
        for seed in exp["seeds"]:
            print(f"📦 Processing Archetype: {exp['name']} (Metric: {exp['metric']}, Seed: {seed})")
            
            reducer = umap.UMAP(
                n_neighbors=exp["neighbors"],       
                min_dist=exp["min_dist"],        
                n_components=2,      
                metric=exp["metric"],     
                random_state=seed      
            )
            embeddings_2d = reducer.fit_transform(embeddings)

            trust_neighbors = min(exp["neighbors"], int(len(embeddings) / 2) - 1)
            trust_score = trustworthiness(embeddings, embeddings_2d, n_neighbors=trust_neighbors)
            info_loss_pct = (1.0 - trust_score) * 100

            for color_col in exp["colors"]:
                current_run += 1
                
                plot_df = base_df.copy()
                plot_df['UMAP X'] = embeddings_2d[:, 0]
                plot_df['UMAP Y'] = embeddings_2d[:, 1]

                umap_settings_title = (
                    f"<b>{MODEL_TYPE}</b> Latent Space Map Archetype: <b>{exp['name']}</b><br>"
                    f"<sub><b>Colored By:</b> {color_col} | <b>n_neighbors:</b> {exp['neighbors']} | "
                    f"<b>min_dist:</b> {exp['min_dist']} | <b>metric:</b> {exp['metric']} | <b>seed:</b> {seed}</sub><br>"
                    f"<sub><i><b>Approximate Information Loss:</b> {info_loss_pct:.1f}%</i></sub>"
                )

                custom_orders = None
                custom_color_map = None
                
                if color_col == 'Angle':
                    custom_orders = {'Angle': ['RT-TRV', 'LT-TRV', 'ML-TRV', 'RT-SAG', 'LT-SAG', 'SAG', 'TBD', 'Unknown']}
                    custom_color_map = {
                        'RT-TRV': "#08418c", 'ML-TRV': '#1f77b4','LT-TRV': '#00efff',
                        'RT-SAG': "#ff1414", 'SAG': "#ff5f24", 'LT-SAG': '#ffaa00', 'TBD': "#000000", 'Unknown': '#bcbd22'
                    }

                elif color_col == 'Diagnosis':
                    custom_color_map = {'Normal': "#2c4ba0", 'Abnormal': '#d62728'}

                elif color_col == 'Frame_Range':
                    custom_color_map = {
                        '0-50 Frames': "#2ca02c",
                        '51-150 Frames': "#ff7f0e",
                        '151+ Frames': "#d62728",
                        'Unknown Frames': "#7f7f7f"
                    }

                elif color_col == 'Study':
                    custom_orders = {
                        'Study': ['study0', 'study1', 'study2', 'study3', 'study4', 'study5', 'study6', 'study7', 'study8', 'study9', 'study10']
                    }
                    custom_color_map = {
                        'study0': "#4e0808", 'study1': "#b41f1f", 'study2': '#ff7f0e', 'study3': "#d9dd22",
                        'study4': "#27d644", 'study5': "#1fbdba", 'study6': "#1a75a7", 'study7': "#4f36de",
                        'study8': "#a830df", 'study9': "#a11d87", 'study10': "#e6759a"
                    }

                fig = px.scatter(
                    plot_df, x='UMAP X', y='UMAP Y', color=color_col, hover_name='Filename',
                    hover_data=[
                        'Diagnosis', 'Study', 'Sweep_Number', 'US_Model', 
                        'US_Generation', 'Photometric_Mode', 'Frame_Range', 'FPS_Range', 'Angle'
                    ], 
                    title=umap_settings_title,
                    category_orders=custom_orders,
                    color_discrete_map=custom_color_map,
                    color_discrete_sequence=px.colors.qualitative.Dark24
                )

                fig.update_layout(
                    template="plotly_white",
                    margin=dict(l=40, r=20, t=95, b=40), 
                    title=dict(
                        text=umap_settings_title,
                        font=dict(size=24, family="Arial Black", color="#111111"),
                        y=0.95,
                        x=0.02
                    ),
                    legend=dict(
                        font=dict(size=20, family="Arial", color="black"),
                        itemsizing="constant", 
                        title=dict(font=dict(size=22, family="Arial Black"))
                    ),
                    xaxis=dict(
                        title=dict(font=dict(size=18, family="Arial Black")),
                        tickfont=dict(size=14)
                    ),
                    yaxis=dict(
                        title=dict(font=dict(size=18, family="Arial Black")),
                        tickfont=dict(size=14)
                    )
                )

                fig.update_traces(marker=dict(size=16, line=dict(width=1.5, color='DarkSlateGrey')))

                export_filename = f"{MODEL_TYPE}_{exp['name']}_grid_{color_col}_n{exp['neighbors']}_s{seed}"

                fig.show(config={
                    'toImageButtonOptions': {
                        'format': 'png', 'filename': export_filename,
                        'height': 720,   
                        'width': 960,    
                        'scale': 3       
                    }
                })

                # Save unique HTML interactive files to target output_dir path structure
                html_path = os.path.join(output_dir, f"{export_filename}.html")
                fig.write_html(html_path, include_plotlyjs='cdn')
                print(f"   [{current_run}/{total_runs}] Saved Grid Map -> {html_path}")

    print(f"\n🎉 Pipeline Complete! Plots generated inside directory:\n -> {output_dir}")

if __name__ == "__main__":
    main()