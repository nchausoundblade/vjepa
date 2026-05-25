import os
import numpy as np
import pandas as pd
import umap
import plotly.express as px

def main():
    # 1. Load the generated matrices
    try:
        embeddings = np.load("/Users/noahchau/Desktop/ultrasound_embeddings.npy")
        labels = np.load("/Users/noahchau/Desktop/ultrasound_labels.npy")
    except FileNotFoundError:
        print("Error: Could not find .npy files. Run 'extract_embeddings.py' first!")
        return

    # 2. Extract filenames from your original CSV for the hover tooltips
    try:
        # Reads the CSV just like your dataset class did
        df_csv = pd.read_csv("vjepa_dataset.csv", sep=r'\s+|,', names=['path', 'label'], engine='python')
        # Isolate just the filename from the full path (e.g., 'sweep_001.mp4')
        filenames = [os.path.basename(p) for p in df_csv['path']]
    except FileNotFoundError:
        print("Error: Could not find 'vjepa_dataset.csv'. Ensure it is in the same folder.")
        return

    # Map the 0 and 1 integers to actual clinical names for the legend
    # (Adjust these strings if your labels mean something different!)
    label_names = ["Abnormal" if label == 1 else "Normal" for label in labels]

    print(f"Loaded {embeddings.shape[0]} embeddings. Computing UMAP projection...")

    # 3. Configure UMAP 
    # CHANGE VARIABLES BELOW TO EXPERIMENT WITH DIFFERENT SETTINGS
    n_neighbors_val = 5
    min_dist_val = 0.0
    n_components_val = 2
    metric_val = 'cosine'
    seed_val = 42   

    print(f"Computing UMAP projection (n_neighbors={n_neighbors_val}, min_dist={min_dist_val})...")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors_val,       
        min_dist=min_dist_val,        
        n_components=n_components_val,      
        metric=metric_val,     
        random_state=seed_val      
    )

    embeddings_2d = reducer.fit_transform(embeddings)

    # 4. Package data into a Pandas DataFrame for Plotly
    plot_df = pd.DataFrame({
        'UMAP X': embeddings_2d[:, 0],
        'UMAP Y': embeddings_2d[:, 1],
        'Diagnosis': label_names,
        'Filename': filenames
    })

    # 5. Generate the Interactive Plot with Dynamic Subtitle Settings
    umap_settings_title = (
        f"Interactive V-JEPA Latent Space (UMAP)<br>"
        f"<sub><b>Hyperparameters:</b> n_neighbors={n_neighbors_val} | "
        f"min_dist={min_dist_val} | metric={metric_val} | random_state={seed_val}</sub>"
    )

    fig = px.scatter(
        plot_df,
        x='UMAP X',
        y='UMAP Y',
        color='Diagnosis',        
        hover_name='Filename',    
        color_discrete_sequence=['#1f77b4', '#d62728'], 
        title=umap_settings_title, # <-- Pass the HTML string here
    )

    # Enhance the visual styling
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        template="plotly_white",
        hoverlabel=dict(bgcolor="white", font_size=14, font_family="Rockwell")
    )

    # 6. Display and Export
    # This will open the interactive plot in your default web browser
    fig.show()
    
    # Also saves an interactive HTML copy to your folder that you can share with colleagues
    fig.write_html("interactive_umap_map.html", include_plotlyjs='cdn')
    print("Exported interactive map to 'interactive_umap_map.html'!")

if __name__ == "__main__":
    main()