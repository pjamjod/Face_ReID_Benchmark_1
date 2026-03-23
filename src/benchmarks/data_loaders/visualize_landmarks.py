"""
Comprehensive visualization of 300W dataset landmarks across all subdatasets
Combines single sample overlays, multi-sample views, and test set comparisons
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from data_loaders.data_loader_300w import Dataset300W

def visualize_landmarks_overlay(sample_idx=0, save_path=None):
    """Visualize both 68-point and 5-point landmarks on a sample image"""

    # Load dataset
    dataset = Dataset300W()

    if len(dataset) == 0:
        print("❌ No images loaded")
        return

    # Get sample
    sample = dataset[sample_idx]
    image = sample['image']
    landmarks_68 = sample['landmarks_68']
    landmarks_5 = sample['landmarks_5']
    image_path = sample['image_path']

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.set_title(f'300W Landmark Overlay - Sample {sample_idx}\n{image_path.split("/")[-1]}')
    ax.axis('off')

    # Plot 68-point landmarks (blue dots)
    ax.scatter(landmarks_68[:, 0], landmarks_68[:, 1],
               c='blue', s=20, alpha=0.7, label='68-point landmarks')

    # Plot 5-point landmarks (red stars)
    ax.scatter(landmarks_5[:, 0], landmarks_5[:, 1],
               c='red', marker='*', s=200, alpha=0.9, label='5-point landmarks')

    # Add labels for 5-point landmarks
    landmark_names = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']
    for i, (point, name) in enumerate(zip(landmarks_5, landmark_names)):
        ax.annotate(name, (point[0], point[1]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Add legend
    ax.legend(loc='upper right')

    # Add statistics
    stats_text = f'Image shape: {image.shape}\n68-pt range: [{landmarks_68.min():.1f}, {landmarks_68.max():.1f}]\n5-pt range: [{landmarks_5.min():.1f}, {landmarks_5.max():.1f}]'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()

def show_multiple_samples(save_path=None):
    """Show overlays for multiple samples"""
    dataset = Dataset300W()

    if len(dataset) < 3:
        print("❌ Need at least 3 samples")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, ax in enumerate(axes):
        sample = dataset[i]
        image = sample['image']
        landmarks_68 = sample['landmarks_68']
        landmarks_5 = sample['landmarks_5']

        ax.imshow(image)
        ax.set_title(f'Sample {i}')
        ax.axis('off')

        # Plot landmarks
        ax.scatter(landmarks_68[:, 0], landmarks_68[:, 1],
                   c='blue', s=10, alpha=0.6)
        ax.scatter(landmarks_5[:, 0], landmarks_5[:, 1],
                   c='red', marker='*', s=150, alpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved multi-sample visualization to: {save_path}")
    else:
        plt.show()

def visualize_test_sets():
    """Create visualizations showing landmarks from different 300W test sets"""

    dataset = Dataset300W()

    if len(dataset) == 0:
        print("❌ No images loaded")
        return

    # Group images by dataset
    datasets = {}
    for i, img_path in enumerate(dataset.image_paths):
        rel_path = os.path.relpath(img_path, "datasets/300w")
        parts = rel_path.split(os.sep)

        if len(parts) >= 2 and parts[0] == "ibug_300W_large_face_landmark_dataset":
            dataset_key = parts[1]
            if len(parts) > 2 and parts[2] in ["testset", "trainset"]:
                dataset_key = f"{parts[1]}_{parts[2]}"
        else:
            dataset_key = parts[0]

        if dataset_key not in datasets:
            datasets[dataset_key] = []
        datasets[dataset_key].append(i)

    # Create output directory
    output_dir = "visualizations_300w_complete"
    os.makedirs(output_dir, exist_ok=True)

    print(f"🖼️  Creating comprehensive visualizations for {len(datasets)} test sets...")

    # Create overview visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    plot_idx = 0
    for dataset_name, indices in sorted(datasets.items()):
        if plot_idx >= 6:  # Limit to 6 subplots
            break

        if len(indices) == 0:
            continue

        # Get first sample from this dataset
        sample_idx = indices[0]
        sample = dataset[sample_idx]
        image = sample['image']
        landmarks_68 = sample['landmarks_68']
        landmarks_5 = sample['landmarks_5']

        ax = axes[plot_idx]
        ax.imshow(image)
        ax.scatter(landmarks_68[:, 0], landmarks_68[:, 1],
                   c='blue', s=8, alpha=0.6, label='68-pt')
        ax.scatter(landmarks_5[:, 0], landmarks_5[:, 1],
                   c='red', marker='*', s=100, alpha=0.9, label='5-pt')

        ax.set_title(f'{dataset_name.upper()}\n{image.shape}')
        ax.axis('off')
        ax.legend(loc='upper right')

        plot_idx += 1

    # Hide empty subplots
    for i in range(plot_idx, 6):
        axes[i].axis('off')

    plt.tight_layout()
    overview_path = os.path.join(output_dir, "testsets_overview.png")
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"💾 Saved overview: {overview_path}")

    # Create detailed visualizations for each test set
    for dataset_name, indices in sorted(datasets.items()):
        if len(indices) < 3:  # Skip small datasets
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i in range(min(3, len(indices))):
            sample_idx = indices[i]
            sample = dataset[sample_idx]
            image = sample['image']
            landmarks_68 = sample['landmarks_68']
            landmarks_5 = sample['landmarks_5']

            ax = axes[i]
            ax.imshow(image)
            ax.scatter(landmarks_68[:, 0], landmarks_68[:, 1],
                       c='cyan', s=15, alpha=0.7)
            ax.scatter(landmarks_5[:, 0], landmarks_5[:, 1],
                       c='red', marker='*', s=150, alpha=0.9)

            # Add labels for 5-point landmarks
            landmark_names = ['Left Eye', 'Right Eye', 'Nose', 'Left Mouth', 'Right Mouth']
            for j, (point, name) in enumerate(zip(landmarks_5, landmark_names)):
                ax.annotate(name.split()[0], (point[0], point[1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='red', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

            ax.set_title(f'Sample {i+1}')
            ax.axis('off')

        fig.suptitle(f'{dataset_name.upper()} Test Set - Landmark Examples', fontsize=14)
        plt.tight_layout()

        detail_path = os.path.join(output_dir, f"{dataset_name}_examples.png")
        plt.savefig(detail_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"💾 Saved {dataset_name}: {detail_path}")

    print(f"\n✅ Visualizations saved in: {output_dir}/")
    print("📁 Created files:")
    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith('.png'):
            print(f"  - {filename}")

def create_comprehensive_visualization():
    """Create all visualizations for the complete 300W dataset"""
    print("🎨 Creating comprehensive 300W dataset visualizations...")

    # Create output directory
    output_dir = "visualizations_300w_complete"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Single sample overlay
    print("📊 Creating single sample overlay...")
    visualize_landmarks_overlay(sample_idx=0,
                               save_path=os.path.join(output_dir, "single_sample_overlay.png"))

    # 2. Multi-sample view
    print("📊 Creating multi-sample view...")
    show_multiple_samples(save_path=os.path.join(output_dir, "multi_sample_view.png"))

    # 3. Test set visualizations
    print("📊 Creating test set visualizations...")
    visualize_test_sets()

    print("\n✅ All visualizations completed!")
    print(f"📁 Output directory: {output_dir}/")

if __name__ == "__main__":
    create_comprehensive_visualization()