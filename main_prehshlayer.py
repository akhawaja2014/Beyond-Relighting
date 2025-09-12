
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from torchsummary import summary
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as path_effects
import itertools
from adjustText import adjust_text
from matplotlib.lines import Line2D


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
        self.preprocessor = HSHPreprocessor()
        self.preprocessor.load_state_dict(torch.load("hsh_preprocessor7.pth", map_location=self.device))
        self.preprocessor.eval()
        self.preprocessor.to(self.device)

    def load_images_from_folder(self, folder_path):
        """
        Load 27-channel HSH tensors from subfolders (e.g., V1A/, P1/).

        Args:
            folder_path (str): Path containing fragment subfolders

        Returns:
            tuple: (fragment folder names, list of HSH tensors)
        """
        fragment_folders = sorted([
            os.path.join(folder_path, d)
            for d in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, d))
        ])

        processed_images = []
        for frag_path in fragment_folders:
            coeff_tensors = []
            for i in range(9):  # H0 to H8
                coeff_file = os.path.join(frag_path, f"plane_{i}.jpg")
                if not os.path.exists(coeff_file):
                    raise FileNotFoundError(f"Missing: {coeff_file}")
                img = Image.open(coeff_file).convert('RGB')
                img_tensor = transforms.ToTensor()(img)  # shape: (3, H, W)
                img_tensor = transforms.Resize((224, 224))(img_tensor)  # ensure uniform size
                coeff_tensors.append(img_tensor)

            hsh_tensor = torch.cat(coeff_tensors, dim=0)  # (27, 224, 224)
            hsh_tensor = (hsh_tensor - hsh_tensor.mean()) / hsh_tensor.std()  # Normalize
            processed_images.append(hsh_tensor.unsqueeze(0).to(self.device))  # Add batch dim

        return fragment_folders, processed_images

    def extract_features(self, image):
        """Extract features from image using the model."""
        with torch.no_grad():
            image = image.to(self.device)  # shape: [1, 27, H, W]
            image = self.preprocessor(image) 
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

        return distance_matrix, features_list


    def visualize_mds(self, distance_matrix, image_paths, n_components=2, output_path=None):
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

      # Perform MDS
      mds = MDS(n_components=n_components,
                dissimilarity='euclidean',
                random_state=42)

      # Fit and transform the distance matrix
      mds_coords = mds.fit_transform(distance_matrix)

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


class HSHPreprocessor(nn.Module):
    def __init__(self, in_channels=27, out_channels=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.block(x)



def main():
    # Specify the folder path containing images
    #folder_path = "HSHplane_0_for_similarity/" 
    #folder_path = "HSHplane0_VP/" 
    #folder_path = "RGBpolish/"  # Replace with your folder path
    folder_path = "hsh_data/"  # Or wherever V1A/, P1/, etc. are stored
    # Create calculator instance
    calculator = FolderImageSimilarityCalculator()

    # Load images from folder
    image_paths, processed_images = calculator.load_images_from_folder(folder_path)

    # Check if any images were found
    if not image_paths:
        print(f"No .jpg images found in the folder: {folder_path}")
        return

    # Calculate similarity matrix
    similarity_matrix, features_list = calculator.calculate_similarity_matrix(processed_images)

    # Print similarity matrix
    print("Similarity Matrix:")
    print(similarity_matrix)

    # Visualize similarity matrix
    calculator.visualize_similarity_matrix(image_paths, similarity_matrix, output_path='RGBV_similarity_matrix_.pdf')

    # Visualize feature space
    calculator.visualize_feature_space(features_list, image_paths, output_path='RGBV_euc_pca_2d.pdf')

    # Calculate and visualize MDS
    distance_matrix, _ = calculator.calculate_distance_matrix(processed_images)

    # Visualize MDS in 2D
    calculator.visualize_mds(distance_matrix, image_paths, n_components=2,  output_path='RGBV_euc_mds_2d.pdf')

    # Uncomment the following line to visualize MDS in 3D
    #calculator.visualize_mds(distance_matrix, image_paths, n_components=3, output_path='RGBP_euc_mds_3d.pdf')

if __name__ == "__main__":
    main()