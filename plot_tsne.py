import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def main():
    # 1. Load the vectors created by the feature extractor script
    try:
        embeddings = np.load("/Users/noahchau/Desktop/ultrasound_embeddings.npy")
        labels = np.load("/Users/noahchau/Desktop/ultrasound_labels.npy")
    except FileNotFoundError:
        print("Error: Could not find embedding files. Run 'extract_embeddings.py' first!")
        return

    print(f"Loaded feature matrix shape: {embeddings.shape}") # Expecting (41, 1024)

    # 2. Configure Scikit-Learn's t-SNE
    # CRITICAL HYPERPARAMETER FOR SMALL DATASETS: Perplexity
    # Default is 30.0, but perplexity MUST be smaller than your total samples (41).
    # Since you have 41 files, a perplexity of 5 to 15 is optimal.
    tsne = TSNE(
        n_components=2,      # Squeeze vectors down to a 2D X/Y coordinate system
        perplexity=5.0,     # Determines balancing of local vs global clusters
        learning_rate='auto',# Auto sets best optimization step sizes
        early_exaggeration=12.0, # Strengthens cluster separation in early optimization
        max_iter=100000000,       # Number of optimization steps to find the best layout
        n_iter_without_progress=300, # Early stopping if no improvement
        min_grad_norm=1e-7,   # Convergence threshold for optimization
        metric='euclidean',    # Distance metric for measuring similarity in high-dimensional space
        metric_params=None,     # Additional parameters for the distance metric (not needed for euclidean)
        init='pca',          # PCA initialization stabilizes global structures 
        verbose=1,           # Set to 1 or 2 for detailed optimization logs
        random_state=42,    # Hardcodes random seed so your plot looks identical every run
        method='barnes_hut',     # Efficient algorithm for large datasets (default for n_samples > 1000)
        angle=0.5,             # Trade-off parameter for Barnes-Hut approximation (lower = more accurate, higher = faster
        n_jobs=None            # Use all CPU cores for parallel computation (speeds up t-SNE on larger datasets
    )

    print("Computing t-SNE 2D mapping layout...")
    embeddings_2d = tsne.fit_transform(embeddings) # Shape turns into (41, 2)

    # 3. Plot the final visual scatter graph
    plt.figure(figsize=(10, 8))
    
    # Generate scatter plots tracking our 2D dimensions
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=labels,            # Colors dots based on normal (0) vs abnormal (1) integers
        cmap='coolwarm',     # Blue-to-Red clinical visualization palette
        s=120,               # Circle point size
        edgecolors='black',  # High contrast edge outlines
        alpha=0.85
    )

    # Chart formatting elements
    plt.title("V-JEPA Latent Space Map (41 Ultrasound Cine Sweeps)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("t-SNE Coordinate 1", fontsize=11)
    plt.ylabel("t-SNE Coordinate 2", fontsize=11)
    
    # Adding clear text indicators for your dataset labels
    # (Adjust text variables below to match your unique clinical definitions)
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.set_ticklabels(['Category 0 (Normal)', 'Category 1 (Abnormal)'], fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Save a high-res image of your map onto disk
    plt.savefig("ultrasound_tsne_map.png", dpi=300)
    print("Map visualization exported successfully as 'ultrasound_tsne_map.png'!")
    
    # Display the popup interactive plot panel
    plt.show()

if __name__ == "__main__":
    main()