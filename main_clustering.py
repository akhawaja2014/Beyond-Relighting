
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
# from torchsummary import summary
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as path_effects
import itertools
from adjustText import adjust_text
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import csv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import trustworthiness
import umap
from sklearn.metrics import pairwise_distances



class FolderImageSimilarityCalculator:
    def __init__(self, model_name='resnet50'):
        # Load pre-trained model
        self.model = getattr(models, model_name)(pretrained=True)
        # Remove the last fully connected layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 3. Define and load the trained HSH preprocessor
        # self.preprocessor = HSHPreprocessor()
        # self.preprocessor.load_state_dict(torch.load("hsh_preprocessor.pth", map_location=self.device))
        # self.preprocessor.eval()
        # self.preprocessor.to(self.device)

    def load_images_from_folder(self, folder_path):
        """
        Load all .jpg images from a given folder.

        Args:
            folder_path (str): Path to the folder containing images

        Returns:
            tuple: (list of image paths, list of processed images)
        """
        # Get all .jpg files in the folder
        image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith('.jpg')
        ]

        # Process images
        processed_images = []
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            processed_image = self.transform(image)
            processed_images.append(processed_image.unsqueeze(0).to(self.device))

        return image_paths, processed_images

    def extract_features(self, image):
        """Extract features from image using the model."""
        with torch.no_grad():
            features = self.model(image)
            features = features.squeeze()
            # Normalize feature vector
            features = F.normalize(features, p=2, dim=0)
        return features

    def calculate_similarity_matrix(self, processed_images):
        """
        Calculate similarity matrix for multiple images.

        Args:
            processed_images (list): List of processed images

        Returns:
            numpy.ndarray: Similarity matrix
            list: List of feature vectors for visualization
        """
        # Extract features for all images
        features_list = []
        for img in processed_images:
            features = self.extract_features(img)
            features_list.append(features)

        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(processed_images), len(processed_images)))
        for i, j in itertools.product(range(len(processed_images)), repeat=2):
            similarity = F.cosine_similarity(
                features_list[i].unsqueeze(0),
                features_list[j].unsqueeze(0)
            ).item()
            similarity_matrix[i, j] = similarity

        return similarity_matrix, features_list

    def visualize_similarity_matrix(self, image_paths, similarity_matrix, output_path=None):
        """
        Visualize the similarity matrix as a heatmap.

        Args:
            image_paths (list): List of image paths
            similarity_matrix (numpy.ndarray): Similarity matrix
        """

        if output_path is None:
          output_path = os.path.join(os.getcwd(), 'similarity_matrix.eps')

        plt.figure(figsize=(10, 10), dpi=400)
        # Use a professional color scheme
        plt.imshow(similarity_matrix, cmap='coolwarm', interpolation='nearest')
        cbar = plt.colorbar(label='Cosine Similarity', fraction=0.046, pad=0.04)
        cbar.ax.tick_labels = [f'{i:.1f}' for i in cbar.get_ticks()]
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.set_ylabel('Cosine Similarity', fontsize=16, rotation=270, labelpad=15)
        # Customize the plot appearance
        #plt.title('Image Similarity Matrix', fontsize=14, pad=20)
        plt.xlabel('Fragment Image', fontsize=16, labelpad=10)
        plt.ylabel('Fragment Image', fontsize=16, labelpad=10)

        # Add text annotations with improved visibility
        for i in range(len(image_paths)):
            for j in range(len(image_paths)):
                color = 'white' if abs(similarity_matrix[i, j]) > 0.9 else 'black'
                plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center',
                        color=color, fontsize=8)

        # Customize tick labels
        plt.xticks(range(len(image_paths)),
                  [os.path.splitext(os.path.basename(path))[0] for path in image_paths],
                  rotation=45, ha='right', fontsize=10)
        plt.yticks(range(len(image_paths)),
                  [os.path.splitext(os.path.basename(path))[0] for path in image_paths],
                  fontsize=10)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        #plt.close()
    
    def visualize_multiple_interclass_boxplots1(self, image_paths_list, similarity_matrices, output_path=None):
        """
        Visualize multiple box plots in the same figure for comparison, showing interclass similarities
        between 'V1' and 'P' fragments for each similarity matrix.

        Args:
            image_paths_list (list): List of image paths for each dataset
            similarity_matrices (list): List of similarity matrices (numpy.ndarray)
            output_path (str): Optional path to save the output figure.
        """
        all_interclass_similarities = []

        # Loop over each dataset
        for idx, similarity_matrix in enumerate(similarity_matrices):
            image_paths = image_paths_list[idx]  # Get corresponding image paths for this dataset

            # 1. Parse image labels
            labels = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
            v1_indices = [i for i, label in enumerate(labels) if label.startswith('V12')]
            p_indices = [i for i, label in enumerate(labels) if label.startswith('P')]

            # 2. Collect interclass similarities for this similarity matrix
            interclass_similarities = [
                similarity_matrix[i, j]
                for i in v1_indices
                for j in p_indices
            ]
            all_interclass_similarities.append(interclass_similarities)

        # 3. Generate boxplot for multiple datasets
        plt.figure(figsize=(8, 6))

        # Create box plots for all similarity matrices
        plt.boxplot(all_interclass_similarities, patch_artist=True)

        # Customize appearance
        plt.title("Interclass Similarity: V103 vs P Fragments", fontsize=14)
        plt.ylabel("Cosine Similarity", fontsize=12)
        # Custom x-axis labels: "RTI" for the first dataset, "RGB" for the second dataset
        plt.xticks(range(1, len(similarity_matrices) + 1), ["RTI", "RGB"], fontsize=10, rotation=45)
        plt.grid(True, linestyle='--', alpha=0.5)

        # Optionally save the figure
        if output_path:
            plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        else:
            plt.show()

        # Return the interclass similarities for each dataset
        return all_interclass_similarities

    def visualize_feature_space(self, features_list, image_paths, output_path=None):
        """
        Visualize feature vectors in 2D using PCA with custom colors for different series.

        Args:
            features_list (list): List of feature vectors
            image_paths (list): List of image paths for labeling
            output_path (str): Path to save the EPS file
        """
        if output_path is None:
            output_path = os.path.join(os.getcwd(), 'RGBfeature_space.eps')

        # Convert feature vectors to numpy
        features = torch.stack(features_list).cpu().numpy()

        # Perform PCA
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features)

        # Calculate explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Print variance information
        for i, var in enumerate(explained_variance_ratio[:10]):
            print(f"PC{i+1}: {var:.4f} ({cumulative_variance[i]:.4f} cumulative)")

        # Create high-quality plot
        plt.figure(figsize=(10, 8), dpi=300)

        # Determine colors based on series pattern in filename
        colors = []

        # Keep track of which series belong to which color groups
        v1_series_indices = []  # V1A to V1F
        v12_series_indices = [] # V12A to V12C
        v103_series_indices = [] # V103 series
        p_series_indices = []
        other_indices = []

        for i, path in enumerate(image_paths):
            filename = os.path.basename(path)
            label = os.path.splitext(filename)[0]

            # Check for V1[A-F] series
            if label.startswith('V1') and len(label) >= 3 and label[2] in 'ABCDEF' and not label.startswith('V12'):
                colors.append('#FFB6C1')  # Crimson for V1A-V1F series
                v1_series_indices.append(i)
            # Check for V12[A-C] series
            elif label.startswith('V12') and len(label) >= 4 and label[3] in 'ABC':
                colors.append('forestgreen')  # Forest green for V12A-V12C series
                v12_series_indices.append(i)
            # Check for V100+ series (numeric patterns)
            elif label.startswith('V103') and len(label) >= 5 and label[4] in 'ABC':
                colors.append('darkorange')  # Orange for V100+ series
                v103_series_indices.append(i)
            elif label.startswith('P') and len(label) >= 2:
                colors.append('#8E44AD')  # Crimson for V1A-V1F series
                p_series_indices.append(i)
            else:
                colors.append('cyan')  # Default color
                other_indices.append(i)

        # Plot points with different colors for different series
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1],
                    c=colors, s=100, alpha=0.6)

        texts = []
        # Add labels with improved positioning
        for i, (x, y) in enumerate(reduced_features):
            filename = os.path.basename(image_paths[i])
            label = os.path.splitext(filename)[0]  # This removes the extension

            # Create annotation and add to texts list
            text = plt.text(x, y, label,
                        fontsize=12,  # Reduced fontsize for less overlap
                        bbox=dict(facecolor='white',
                                  edgecolor='none',
                                  alpha=0.15,
                                  pad=2),  # Added padding
                        ha='left',  # Center align text
                        va='bottom',
                        zorder=10)
            texts.append(text)

        adjust_text(texts,
                  arrowprops=dict(arrowstyle='-', color='gray', lw=1, connectionstyle='arc3,rad=0.1', zorder=0, alpha = 0.3),
                  expand_points=(330, 330),       # Increased expansion
                  force_points=(0.9, 0.9),        # Increased force
                  force_text=(1, 1),          # Added text force
                  lim=1000,                        # Increased iterations
                  add_objects=[scatter],
                  only_move={'points':'y', 'texts':'xy'},
                  autoalign='xy',
                  text_from_points=True,  # Consider distance from original points
                  text_from_text=True,
                  min_arrow_dist=205,
                  )


        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='V1 Series',
                  markerfacecolor='#FFB6C1', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='V12 Series',
                  markerfacecolor='forestgreen', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='V103 Series',
                  markerfacecolor='darkorange', markersize=10),
            #Line2D([0], [0], marker='o', color='w', label='P Series',
             #     markerfacecolor='#8E44AD', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Other Fragments',
                  markerfacecolor='cyan', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc='upper left')

        # Customize plot appearance
        plt.xlabel('Principal Component 1', fontsize=16, labelpad=10)
        plt.ylabel('Principal Component 2', fontsize=16, labelpad=10)

        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.3)

        # Save high-quality EPS
        plt.tight_layout()
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)

    def visualize_feature_space_kmeans(self, features_list, image_paths, cluster_labels=None, output_path=None):
        """
        Visualize feature vectors in 2D using PCA.
        If cluster_labels are provided, color by cluster; otherwise, color by known series.

        Args:
            features_list (list of torch.Tensor): Feature vectors
            image_paths (list of str): Corresponding image filenames
            cluster_labels (list or np.ndarray, optional): Cluster assignments for each image
            output_path (str): Path to save the figure
        """
        if output_path is None:
            output_path = os.path.join(os.getcwd(), 'feature_space_pca.pdf')

        # Convert to NumPy
        features = torch.stack(features_list).cpu().numpy()

        # Reduce to 2D
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(features)

        print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}")

        # Plot
        plt.figure(figsize=(10, 8), dpi=300)

        if cluster_labels is not None:
            # Use cluster labels for color
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1],
                                c=cluster_labels, cmap='tab10', s=100, alpha=0.7)
            cbar = plt.colorbar(scatter, ticks=np.unique(cluster_labels))
            cbar.set_label("Cluster", fontsize=12)
        else:
            # Color manually by image filename prefixes
            colors = []
            for path in image_paths:
                name = os.path.splitext(os.path.basename(path))[0]
                if name.startswith('V1') and len(name) >= 3 and name[2] in 'ABCDEF' and not name.startswith('V12'):
                    colors.append('#FFB6C1')  # Pink: V1A–V1F
                elif name.startswith('V12') and len(name) >= 4 and name[3] in 'ABC':
                    colors.append('forestgreen')  # Green: V12A–V12C
                elif name.startswith('V103') and len(name) >= 5 and name[4] in 'ABC':
                    colors.append('darkorange')  # Orange: V103A–C
                elif name.startswith('P'):
                    colors.append('#8E44AD')  # Purple: P series
                else:
                    colors.append('cyan')  # Default

            scatter = plt.scatter(reduced[:, 0], reduced[:, 1],
                                c=colors, s=100, alpha=0.7)

            # Legend for manual coloring
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='V1 Series', markerfacecolor='#FFB6C1', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='V12 Series', markerfacecolor='forestgreen', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='V103 Series', markerfacecolor='darkorange', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='P Series', markerfacecolor='#8E44AD', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor='cyan', markersize=10)
            ]
            plt.legend(handles=legend_elements, loc='upper right')

        # Labels and styling
        plt.xlabel('Principal Component 1', fontsize=14)
        plt.ylabel('Principal Component 2', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def calculate_distance_matrix(self, processed_images):
        """
        Calculate distance matrix for multiple images.

        Args:
            processed_images (list): List of processed images

        Returns:
            tuple: (distance matrix, feature vectors)
        """
        # Extract features for all images
        features_list = []
        for img in processed_images:
            features = self.extract_features(img)
            features_list.append(features)

        # Convert features to numpy
        features_array = torch.stack(features_list).cpu().numpy()

        # Calculate cosine distance matrix (1 - cosine similarity)
        distance_matrix = np.zeros((len(processed_images), len(processed_images)))
        for i, j in itertools.product(range(len(processed_images)), repeat=2):
            cosine_sim = F.cosine_similarity(
                torch.from_numpy(features_array[i]).unsqueeze(0),
                torch.from_numpy(features_array[j]).unsqueeze(0)
            ).item()
            # Convert cosine similarity to distance (0 is similar, 2 is dissimilar)
            distance_matrix[i, j] = 1 - cosine_sim

        return distance_matrix, features_array

    def visualize_mds(self, distance_matrix, features_list, image_paths, n_components=2, output_path=None):
      """
      Perform Multi-Dimensional Scaling and visualize results with custom colors for different series.

      Args:
          distance_matrix (numpy.ndarray): Distance matrix between images
          image_paths (list): List of image paths
          n_components (int): Number of dimensions to reduce to (2 or 3)
          output_path (str): Path to save the output file
      """
      if output_path is None:
          output_path = os.path.join(os.getcwd(), 'RGBmds_visualization.eps')
      

      features_array = torch.stack(features_list).cpu().numpy()  # shape (n_samples, n_features)

      # Perform MDS
      mds = MDS(n_components= n_components,
                dissimilarity='euclidean',
                random_state=42)

      # Fit and transform the distance matrix
      
      # mds_coords = mds.fit_transform(distance_matrix)
      mds_coords = mds.fit_transform(features_array)

      D_original = pairwise_distances(features_array, metric='euclidean')
      D_fit = pairwise_distances(mds_coords, metric='euclidean')
      # D_fit = pairwise_distances(mds_coords)

      # Kruskal's Stress-1
      stress1 = np.linalg.norm(D_original - D_fit) / np.linalg.norm(D_original)
      print("Normalized Stress-1:", stress1)
      print(f'MDS_Stress: {mds.stress_}')


      # Create high-quality plot
      plt.figure(figsize=(10, 8), dpi=300)

      if n_components == 2:
          # Determine colors based on series pattern in filename
          colors = []

          # Keep track of which series belong to which color groups
          v1_series_indices = []  # V1A to V1F
          v12_series_indices = [] # V12A to V12C
          v103_series_indices = [] 
          p_series_indices = [] # V103 series
          other_indices = [] 

          for i, path in enumerate(image_paths):
              filename = os.path.basename(path)
              label = os.path.splitext(filename)[0]

              # Check for V1[A-F] series
              if label.startswith('V1') and len(label) >= 3 and label[2] in 'ABCDEF' and not label.startswith('V12'):
                  colors.append('#FFB6C1')  # Crimson for V1A-V1F series
                  v1_series_indices.append(i)
              # Check for V12[A-C] series
              elif label.startswith('V12') and len(label) >= 4 and label[3] in 'ABC':
                  colors.append('forestgreen')  # Forest green for V12A-V12C series
                  v12_series_indices.append(i)
              # Check for V103 series
              elif label.startswith('V103') and len(label) >= 5 and label[4] in 'ABC':
                  colors.append('darkorange')  # Orange for V103 series
                  v103_series_indices.append(i)
              elif label.startswith('P') and len(label) >= 2:
                  colors.append('#8E44AD')  # Crimson for V1A-V1F series
                  p_series_indices.append(i)
              else:
                  colors.append('cyan')  # Default color
                  other_indices.append(i)

          # 2D plot with colored points
          scatter = plt.scatter(mds_coords[:, 0], mds_coords[:, 1],
                            c=colors, s=100, alpha=0.6)

          plt.xlabel('MDS Dimension 1', fontsize=16, labelpad=10)
          plt.ylabel('MDS Dimension 2', fontsize=16, labelpad=10)

          texts = []
          # Annotate points with filenames
          for i, (x, y) in enumerate(mds_coords):
              filename = os.path.basename(image_paths[i])
              label = os.path.splitext(filename)[0]
              text = plt.text(x, y, label,
                      fontsize=12,  # Reduced fontsize for less overlap
                      bbox=dict(facecolor='white',
                                edgecolor='none',
                                alpha=0.15,
                                pad=2),  # Added padding
                      ha='left',  # Center align text
                      va='bottom',
                      zorder=10)
              texts.append(text)

          adjust_text(texts,
                arrowprops=dict(arrowstyle='-', color='gray', lw=1, connectionstyle='arc3,rad=0.1', zorder=0, alpha=0.3),
                expand_points=(330, 330),       # Increased expansion
                force_points=(0.9, 0.9),        # Increased force
                force_text=(1, 1),          # Added text force
                lim=1000,                        # Increased iterations
                add_objects=[scatter],
                only_move={'points':'y', 'texts':'xy'},
                autoalign='xy',
                text_from_points=True,  # Consider distance from original points
                text_from_text=True,
                min_arrow_dist=205,
                )

          # Create a custom legend

          legend_elements = [
              Line2D([0], [0], marker='o', color='w', label='V1 Series',
                    markerfacecolor='#FFB6C1', markersize=10),
              Line2D([0], [0], marker='o', color='w', label='V12 Series',
                    markerfacecolor='forestgreen', markersize=10),
              Line2D([0], [0], marker='o', color='w', label='V103 Series',
                    markerfacecolor='darkorange', markersize=10),
             #Line2D([0], [0], marker='o', color='w', label='P Series',
              #      markerfacecolor='#8E44AD', markersize=10),
              Line2D([0], [0], marker='o', color='w', label='Other Fragments',
                    markerfacecolor='cyan', markersize=10)
          ]
          plt.legend(handles=legend_elements, loc='upper left')

          # Add grid for better readability
          plt.grid(True, linestyle='--', alpha=0.3)

          plt.tight_layout()
          plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)

      elif n_components == 3:
          from mpl_toolkits.mplot3d import Axes3D

          # Determine colors based on series pattern in filename
          colors = []
          for i, path in enumerate(image_paths):
              filename = os.path.basename(path)
              label = os.path.splitext(filename)[0]

              # Check for V1[A-F] series
              if label.startswith('V1') and len(label) >= 3 and label[2] in 'ABCDEF' and not label.startswith('V12'):
                  colors.append('#FFB6C1')  # Crimson for V1A-V1F series
              # Check for V12[A-C] series
              elif label.startswith('V12') and len(label) >= 4 and label[3] in 'ABC':
                  colors.append('forestgreen')  # Forest green for V12A-V12C series
              # Check for V103 series
              elif label.startswith('V103') and len(label) >= 5 and label[4] in 'ABC':
                  colors.append('darkorange')  # Orange for V103 series
              elif label.startswith('P') and len(label) >= 2:
                  colors.append('#8E44AD')  # Crimson for V1A-V1F series
              else:
                  colors.append('cyan')  # Default color

          # Increase figure size and ensure enough space for labels
          plt.figure(figsize=(10, 8), dpi=300)  # Increased figure size

          # Create 3D axis with more space for labels
          ax = plt.subplot(111, projection='3d')
          ax.dist = 12  # Increase distance between plot and camera

          scatter = ax.scatter(mds_coords[:, 0], mds_coords[:, 1], mds_coords[:, 2],
                            c=colors, s=100, alpha=0.6)

          # Set labels with increased padding
          ax.set_xlabel('MDS Dimension 1', fontsize=14, labelpad=20)
          ax.set_ylabel('MDS Dimension 2', fontsize=14, labelpad=20)
          ax.set_zlabel('MDS Dimension 3', fontsize=14, labelpad=20)

          # Adjust the viewing angle to better show all labels
          ax.view_init(elev=28, azim=28)  # Adjust these angles as needed3333333

          # Increase tick label sizes
          ax.tick_params(axis='x', which='major', labelsize=12)
          ax.tick_params(axis='y', which='major', labelsize=12)
          ax.tick_params(axis='z', which='major', labelsize=12)

          # Annotate points with filenames
          for i, (x, y, z) in enumerate(mds_coords):
              filename = os.path.basename(image_paths[i])
              label = os.path.splitext(filename)[0]
              ax.text(x, y, z, label,
                      fontsize=7,
                      bbox=dict(facecolor='white',
                              edgecolor='none',
                              alpha=0.15,
                              pad=1))



          # Create a custom legend

          legend_elements = [
              Line2D([0], [0], marker='o', color='w', label='V1 Series',
                    markerfacecolor='#FFB6C1', markersize=10),
              Line2D([0], [0], marker='o', color='w', label='V12 Series',
                    markerfacecolor='forestgreen', markersize=10),
              Line2D([0], [0], marker='o', color='w', label='V103 Series',
                    markerfacecolor='darkorange', markersize=10),
              # Line2D([0], [0], marker='o', color='w', label='P Series',
              #     markerfacecolor='#8E44AD', markersize=10),
              Line2D([0], [0], marker='o', color='w', label='Other Fragments',
                    markerfacecolor='cyan', markersize=10)
          ]
          ax.legend(handles=legend_elements, loc='upper left')

          # Add space around the plot before saving
          plt.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.9)

          # Save with increased padding
          plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight',
                      transparent=True, pad_inches=0.2)  # Increased padding

    def cluster_and_evaluate(self, features_list, dataset_name, max_k=10,save_plot=True, plot_path = None):
        """
        Try K-Means clustering with different k values and evaluate with metrics.

        Args:
            features_list (list of torch.Tensor): List of 2048-d feature vectors
            max_k (int): Maximum number of clusters to try

        Returns:
            dict: Best clustering result and associated scores
        """
        features_array = torch.stack(features_list).cpu().numpy()

        #  Apply PCA before clustering
        pca = PCA(n_components = 4, random_state=42)
        features_array = pca.fit_transform(features_array)


        # Print variance explained by each component
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()

        print("Explained variance by each principal component:")
        for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
            print(f"PC{i+1}: {var:.4f} ({cum_var:.4f} cumulative)")

        # Plot
        # plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=cluster_labels)
        # plt.title("t-SNE of Feature Vectors")
        # plt.show()

        results = []
        for k in range(4, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features_array)

            silhouette = silhouette_score(features_array, labels)
            ch_score = calinski_harabasz_score(features_array, labels)
            db_score = davies_bouldin_score(features_array, labels)

            results.append({
                'k': k,
                'labels': labels,
                'silhouette': silhouette,
                'calinski_harabasz': ch_score,
                'davies_bouldin': db_score
            })


        # Plot silhouette scores
        ks = [r['k'] for r in results]
        silhouettes = [r['silhouette'] for r in results]
        ch_scores = [r['calinski_harabasz'] for r in results]
        db_scores = [r['davies_bouldin'] for r in results]

        plt.figure(figsize=(8, 5))
        plt.plot(ks, silhouettes, marker='o', linestyle='-', color='blue')
        # plt.title("Silhouette Score vs. Number of Clusters (K)")
        plt.xlabel("Number of Clusters (k)",fontsize=24)
        plt.ylabel("Silhouette Score", fontsize=24)
        plt.xticks(ks, fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(plot_path,f'VPsilhouette_vs_k_{dataset_name}.pdf'), format='pdf', dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)


        # Plot 2: Calinski–Harabasz Score
        plt.figure(figsize=(8, 5))
        plt.plot(ks, ch_scores, marker='o', linestyle='-', color='green')
        plt.xlabel("Number of Clusters (k)", fontsize=24)
        plt.ylabel("Calinski–Harabasz Score", fontsize=24)
        #plt.title("CH Score vs. k")
        plt.xticks(ks, fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f'VPch_score_vs_k_{dataset_name}.pdf'), format='pdf', dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)
        # Sort by best silhouette score


        best = max(results, key=lambda x: x['silhouette'])


        # Plot 3: Davies–Bouldin Score
        plt.figure(figsize=(8, 5))
        plt.plot(ks, db_scores, marker='o', linestyle='-', color='red')
        plt.xlabel("Number of Clusters (k)", fontsize=24)
        plt.ylabel("Davies–Bouldin Score", fontsize=24)
        #plt.title("DB Score vs. k")
        plt.xticks(ks, fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f'VPdb_score_vs_k_{dataset_name}.pdf'), format='pdf', dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)

        print(f"Plots saved in: {plot_path}")

        print("\n--- Clustering Evaluation ---")
        for r in results:
            print(f"k={r['k']} | Silhouette={r['silhouette']:.3f} | CH={r['calinski_harabasz']:.1f} | DB={r['davies_bouldin']:.2f}")

        print(f"\n>> Best clustering (by silhouette): k={best['k']}, score={best['silhouette']:.3f}")
        return best

    def visualize_tsne(self, features_list, image_paths, output_path=None, perplexity=20, random_state=42):
        """
        Visualize high-dimensional features using t-SNE.

        Args:
            features_list (list): List of feature vectors (torch.Tensor)
            image_paths (list): Corresponding image paths
            output_path (str): Optional path to save the plot
            perplexity (int): t-SNE perplexity (typically between 5 and 50)
            random_state (int): Random seed for reproducibility
        """

        if output_path is None:
            output_path = os.path.join(os.getcwd(), 'tsne_feature_space.pdf')

        # Convert to numpy array
        features_array = torch.stack(features_list).cpu().numpy()

        # Run t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init='random')
        tsne_coords = tsne.fit_transform(features_array)

        # Color coding
        colors = []
        v1_series_indices = []
        v12_series_indices = []
        v103_series_indices = []
        p_series_indices = []
        other_indices = []

        for i, path in enumerate(image_paths):
            label = os.path.splitext(os.path.basename(path))[0]
            if label.startswith('V1') and len(label) >= 3 and label[2] in 'ABCDEF' and not label.startswith('V12'):
                colors.append('#FFB6C1')
                v1_series_indices.append(i)
            elif label.startswith('V12') and len(label) >= 4 and label[3] in 'ABC':
                colors.append('forestgreen')
                v12_series_indices.append(i)
            elif label.startswith('V103') and len(label) >= 5 and label[4] in 'ABC':
                colors.append('darkorange')
                v103_series_indices.append(i)
            elif label.startswith('P'):
                colors.append('#8E44AD')
                p_series_indices.append(i)
            else:
                colors.append('cyan')
                other_indices.append(i)

        # Plotting
        plt.figure(figsize=(10, 8), dpi=300)
        scatter = plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=colors, s=100, alpha=0.6)

        # Add labels
        texts = []
        for i, (x, y) in enumerate(tsne_coords):
            label = os.path.splitext(os.path.basename(image_paths[i]))[0]
            text = plt.text(x, y, label,
                            fontsize=10,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.15, pad=2),
                            ha='left', va='bottom')
            texts.append(text)

        adjust_text(
            texts,
            arrowprops=dict(arrowstyle='-', color='gray', lw=1, connectionstyle='arc3,rad=0.1', alpha=0.3),
            expand_points=(330, 330),
            force_points=(0.9, 0.9),
            force_text=(1, 1),
            lim=1000,
            add_objects=[scatter],
            only_move={'points': 'y', 'texts': 'xy'},
            autoalign='xy',
            text_from_points=True,
            text_from_text=True,
            min_arrow_dist=205,
        )

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='V1 Series', markerfacecolor='#FFB6C1', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='V12 Series', markerfacecolor='forestgreen', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='V103 Series', markerfacecolor='darkorange', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='P Series', markerfacecolor='#8E44AD', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Other Fragments', markerfacecolor='cyan', markersize=10),
        ]
        plt.legend(handles=legend_elements, loc='lower left')

        plt.title('t-SNE Visualization of Feature Vectors', fontsize=16)
        plt.xlabel('t-SNE Dimension 1', fontsize=14)
        plt.ylabel('t-SNE Dimension 2', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)

        score = trustworthiness(features_array, tsne_coords, n_neighbors=5)
        print(f"Trustworthiness: {score:.4f}")

    def visualize_umap(self, features_list, image_paths, output_path=None, n_neighbors=15, min_dist=0.1, random_state=42):
        """
        Visualize feature vectors using UMAP in 2D with color-coded labels.

        Args:
            features_list (list): List of 2048-d feature vectors (torch.Tensor)
            image_paths (list): Corresponding image paths
            output_path (str): Optional path to save the UMAP plot
            n_neighbors (int): Controls local structure preservation (5–50 typical)
            min_dist (float): Controls how tightly UMAP packs points (0.0–0.5)
            random_state (int): Seed for reproducibility
        """

        if output_path is None:
            output_path = 'umap_feature_space.pdf'

        # Convert to NumPy array
        features_array = torch.stack(features_list).cpu().numpy()

        # Fit UMAP
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        umap_coords = reducer.fit_transform(features_array)

        # Assign colors based on filename patterns
        colors = []
        for i, path in enumerate(image_paths):
            label = os.path.splitext(os.path.basename(path))[0]
            if label.startswith('V1') and len(label) >= 3 and label[2] in 'ABCDEF' and not label.startswith('V12'):
                colors.append('#FFB6C1')  # Pink
            elif label.startswith('V12') and len(label) >= 4 and label[3] in 'ABC':
                colors.append('forestgreen')
            elif label.startswith('V103') and len(label) >= 5 and label[4] in 'ABC':
                colors.append('darkorange')
            elif label.startswith('P'):
                colors.append('#8E44AD')
            else:
                colors.append('cyan')

        # Create plot
        plt.figure(figsize=(10, 8), dpi=300)
        scatter = plt.scatter(umap_coords[:, 0], umap_coords[:, 1], c=colors, s=100, alpha=0.6)

        # Add text labels
        texts = []
        for i, (x, y) in enumerate(umap_coords):
            label = os.path.splitext(os.path.basename(image_paths[i]))[0]
            text = plt.text(x, y, label,
                            fontsize=10,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.2, pad=2),
                            ha='left', va='bottom', zorder=10)
            texts.append(text)

        adjust_text(texts,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=1, alpha=0.3),
                    expand_points=(330, 330),
                    force_points=(0.9, 0.9),
                    force_text=(1, 1),
                    lim=1000,
                    add_objects=[scatter],
                    only_move={'points': 'y', 'texts': 'xy'},
                    autoalign='xy',
                    text_from_points=True,
                    min_arrow_dist=205)

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='V1 Series', markerfacecolor='#FFB6C1', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='V12 Series', markerfacecolor='forestgreen', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='V103 Series', markerfacecolor='darkorange', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Other Fragments', markerfacecolor='cyan', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc='lower left')

        # Final touches
        plt.xlabel('UMAP Dimension 1', fontsize=14)
        plt.ylabel('UMAP Dimension 2', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)

        score = trustworthiness(features_array, umap_coords, n_neighbors=n_neighbors)
        print(f"TrustworthinessUMAP: {score:.3f}")

def main():
    # Specify the folder path containing images
    #RTI_folder_path = "HSHplane_0_for_similarity/" 
    RTI_VP_folder_path = "HSHplane_0_V/" 
    # RTI_VP_folder_path = "HSHplane_0_V/" 
    #folder_path = "RGBpolish/"  # Replace with your folder path
    RGB_VP_folder_path = "RGB_V/"
    #RGB_VP_folder_path = "RGB_V/"
    # Create calculator instance
    RTI_sim = FolderImageSimilarityCalculator()
    RGB_sim = FolderImageSimilarityCalculator()
    
    # Load images from folder
    image_paths_RTI, processed_images_RTI = RTI_sim.load_images_from_folder(RTI_VP_folder_path)
    image_paths_RGB, processed_images_RGB = RGB_sim.load_images_from_folder(RGB_VP_folder_path)
    
    # Check if any images were found
    if not image_paths_RTI:
        print(f"No .jpg images found in the RTI folder: {RTI_VP_folder_path}")
        return
    if not image_paths_RGB:
        print(f"No .jpg images found in the RGB folder: {RGB_VP_folder_path}")
        return

    # Calculate similarity matrix
    RTI_similarity_matrix, RTI_features_list = RTI_sim.calculate_similarity_matrix(processed_images_RTI)
    RGB_similarity_matrix, RGB_features_list = RGB_sim.calculate_similarity_matrix(processed_images_RGB)
    
    
    # Print similarity matrix
    # print("Similarity Matrix:")
    # print(similarity_matrix)
    
    # Visualize similarity matrix
    # RTI_sim.visualize_similarity_matrix(image_paths_RTI, RTI_similarity_matrix, output_path='figures/RTI_VP_similarity_matrix_.pdf')
    # RGB_sim.visualize_similarity_matrix(image_paths_RGB, RGB_similarity_matrix, output_path='figures/RGB_VP_similarity_matrix_.pdf')

    similarity_matrices = [RTI_similarity_matrix, RGB_similarity_matrix]
    image_paths = [image_paths_RTI, image_paths_RGB]
    # interclass_similarities = RGB_sim.visualize_multiple_interclass_boxplots1(image_paths, similarity_matrices, output_path='Test_RGBV_similarity_matrix.pdf')


    # Try clustering with different k values and get best by silhouette
    best_result_RTI = RTI_sim.cluster_and_evaluate(RTI_features_list, max_k=10,dataset_name ="RTI", plot_path="figures/figuresRTI")
    RTI_cluster_labels = best_result_RTI['labels']

    best_result_RGB = RGB_sim.cluster_and_evaluate(RGB_features_list, max_k=10, dataset_name ="RGB", plot_path="figures/figuresRGB")
    RGB_cluster_labels = best_result_RGB['labels']

    # Map image names to their cluster assignments
    # for cluster_id in sorted(set(RTI_cluster_labels)):
    #     print(f"\nCluster {cluster_id}:")
    #     for i, label in enumerate(RTI_cluster_labels):
    #         if label == cluster_id:
    #             print(f"  - {os.path.basename(image_paths_RTI[i])}")



    # with open("cluster_assignments_RTI_VP_PCA4Kmeans.csv", "w", newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["RTI Image", "Cluster"])
    #     for i, label in enumerate(RTI_cluster_labels):
    #         writer.writerow([os.path.basename(image_paths_RTI[i]), label])






    # Map image names to their cluster assignments
    # for cluster_id in sorted(set(RGB_cluster_labels)):
    #     print(f"\nCluster {cluster_id}:")
    #     for i, label in enumerate(RGB_cluster_labels):
    #         if label == cluster_id:
    #             print(f"  - {os.path.basename(image_paths_RGB[i])}")



    # with open("cluster_assignments_RGB_VP_PCA4Kmeans.csv", "w", newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["RGB Image", "Cluster"])
    #     for i, label in enumerate(RGB_cluster_labels):
    #         writer.writerow([os.path.basename(image_paths_RGB[i]), label])


    RTI_sim.visualize_tsne(RTI_features_list, image_paths_RTI, output_path='figures/RTIVP_tsne.pdf', perplexity=5)
    RGB_sim.visualize_tsne(RGB_features_list, image_paths_RGB, output_path='figures/RGBVP_tsne.pdf', perplexity=5)

    print("Thank you for your patience")
    RTI_sim.visualize_umap(RTI_features_list, image_paths_RTI, output_path='figures/rtiVP_umap.pdf', n_neighbors=5, min_dist=0.5)
    RGB_sim.visualize_umap(RGB_features_list, image_paths_RGB, output_path='figures/rgbVP_umap.pdf', n_neighbors=5, min_dist=0.5)   
    # Visualize clusters with PCA
    # RTI_sim.visualize_feature_space_kmeans(RTI_features_list, image_paths_RTI, cluster_labels=RTI_cluster_labels, output_path='RTI_clusters_pca.pdf')

    # Visualize feature space PCA
    RTI_sim.visualize_feature_space(RTI_features_list, image_paths_RTI, output_path='figures/RTIVP_pca_2d.pdf')
    RGB_sim.visualize_feature_space(RGB_features_list, image_paths_RGB, output_path='figures/RGBVP_pca_2d.pdf')
    # Calculate and visualize MDS
    distance_matrix, _ = RGB_sim.calculate_distance_matrix(processed_images_RGB)

    # Visualize MDS in 2D
    RTI_sim.visualize_mds(distance_matrix, RTI_features_list, image_paths_RTI, n_components=2,  output_path='figures/RTIVP_euc_mds_2d.pdf')
    RGB_sim.visualize_mds(distance_matrix,  RGB_features_list, image_paths_RGB, n_components=2,  output_path='figures/RGBVP_euc_mds_2d.pdf')
    # Uncomment the following line to visualize MDS in 3D
    #calculator.visualize_mds(distance_matrix, image_paths, n_components=3, output_path='RGBP_euc_mds_3d.pdf')

if __name__ == "__main__":
    main()