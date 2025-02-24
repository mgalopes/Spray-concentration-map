import cv2
import numpy as np
import os
from pathlib import Path
import logging
from typing import List, Optional
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageProcessor:
    def __init__(self, 
                 subtracted_root: str,
                 output_root: str,
                 threshold: Optional[int] = None,
                 adaptive_threshold: bool = True,
                 gaussian_sigma: float = 0.4,
                 colormap: int = cv2.COLORMAP_JET,
                 log_scale: bool = True,
                 remove_vertical_noise: bool = True,
                 median_kernel_size: int = 3,
                 morph_kernel_size: int = 5,
                 max_distance: int = 50,  # Maximum allowed distance from main object
                 min_object_area: int = 100,  # Minimum area to consider an object
                 noise_threshold: float = 0.8):  # Frequency threshold for noise
        """
        Enhanced image processor with noise filtering.
        
        Parameters:
            subtracted_root (str): Root directory containing subtracted images
            output_root (str): Root directory to save colormaps
            threshold (int, optional): Fixed threshold value (if not using adaptive)
            adaptive_threshold (bool): Use adaptive Otsu thresholding
            gaussian_sigma (float): Sigma for Gaussian blur preprocessing
            colormap: OpenCV colormap to use
            log_scale: Apply logarithmic scaling to frequency map
            remove_vertical_noise (bool): Enable vertical noise removal
            median_kernel_size (int): Kernel size for median filtering
            morph_kernel_size (int): Kernel size for morphological operations
            max_distance (int): Maximum allowed distance from the main object (in pixels)
            min_object_area (int): Minimum area to consider an object
            noise_threshold (float): Frequency threshold for noise removal
        """
        self.subtracted_root = subtracted_root
        self.output_root = output_root
        self.threshold = threshold
        self.adaptive_threshold = adaptive_threshold
        self.gaussian_sigma = gaussian_sigma
        self.colormap = colormap
        self.log_scale = log_scale
        self.remove_vertical_noise = remove_vertical_noise
        self.median_kernel_size = median_kernel_size
        self.morph_kernel_size = morph_kernel_size
        self.max_distance = max_distance
        self.min_object_area = min_object_area
        self.noise_threshold = noise_threshold
        
        # Validate parameters
        if not adaptive_threshold and threshold is None:
            raise ValueError("Either set adaptive_threshold=True or provide a threshold value")
        
        os.makedirs(self.output_root, exist_ok=True)

        # Add processing counters
        self.total_dirs = 0
        self.processed_dirs = 0
        self.start_time = None

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing steps including Gaussian blur"""
        blurred = cv2.GaussianBlur(image, (0, 0), self.gaussian_sigma)
        return blurred

    def _binarize_image(self, image_path: str) -> Optional[np.ndarray]:
        """Advanced binarization with adaptive thresholding"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
                
            preprocessed = self._preprocess_image(image)
            
            if self.adaptive_threshold:
                # Otsu's thresholding with automatic value
                threshold, binarized = cv2.threshold(
                    preprocessed, 0, 255, 
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                logging.debug(f"Otsu's threshold for {image_path}: {threshold}")
            else:
                _, binarized = cv2.threshold(
                    preprocessed, self.threshold, 255, cv2.THRESH_BINARY
                )
            
            return binarized
        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
            return None

    def _remove_vertical_noise(self, image: np.ndarray) -> np.ndarray:
        """Remove vertical noise using morphological operations and median filtering"""
        # Convert to uint8 for OpenCV operations
        image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Median filtering to reduce noise
        denoised = cv2.medianBlur(image_8bit, self.median_kernel_size)
        
        # Morphological opening to remove vertical lines
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, self.morph_kernel_size)
        )
        opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, vertical_kernel)
        
        return opened.astype(np.float32) / 255.0  # Preserve float32 precision

    def _find_main_object(self, binary_image: np.ndarray) -> Optional[tuple]:
        """Identify the largest object and return its centroid and mask"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image.astype(np.uint8), connectivity=8
        )
        if num_labels < 2:
            return None, None
        
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        main_mask = (labels == largest_label).astype(np.uint8) * 255
        M = cv2.moments(main_mask)
        if M["m00"] == 0:
            return None, main_mask
            
        centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return centroid, main_mask

    def _filter_distant_objects(self, binary_image: np.ndarray, frequency_map: np.ndarray) -> np.ndarray:
        """Enhanced noise filtering with edge cases handling"""
        # Convert frequency map to 0-1 range
        freq_normalized = cv2.normalize(frequency_map, None, 0, 1, cv2.NORM_MINMAX)
        
        main_centroid, main_mask = self._find_main_object(binary_image)
        if main_centroid is None:
            return binary_image  # No main object found

        contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_mask = np.zeros_like(binary_image)

        for contour in contours:
            if cv2.contourArea(contour) < self.min_object_area:
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            distance = np.linalg.norm(np.array(centroid) - np.array(main_centroid))
            frequency = freq_normalized[centroid[1], centroid[0]]

            if distance <= self.max_distance or frequency < self.noise_threshold:
                cv2.drawContours(valid_mask, [contour], -1, 255, thickness=cv2.FILLED)

        return valid_mask

    def _process_directory(self, dir_path: str, files: List[str]) -> Optional[str]:
        """Process all images in a single directory"""
        dir_name = Path(dir_path).name
        print(f"\n● Processing directory: {dir_name}")
        print(f"  └ Found {len(files)} images")
        
        # Initialize frequency map using first image dimensions
        first_image = cv2.imread(os.path.join(dir_path, files[0]), cv2.IMREAD_GRAYSCALE)
        if first_image is None:
            print(f"  ⚠ Warning: Could not read the first image in {dir_name}")
            return None
            
        frequency_map = np.zeros(first_image.shape, dtype=np.float32)
        valid_images = 0

        for filename in files:
            image_path = os.path.join(dir_path, filename)
            binarized = self._binarize_image(image_path)
            
            if binarized is not None:
                if binarized.shape != frequency_map.shape:
                    print(f"  ⚠ Warning: Image {filename} has different dimensions. Skipping.")
                    continue
                    
                frequency_map += binarized / 255.0  # Accumulate as float32
                valid_images += 1

        if valid_images == 0:
            print(f"  ⚠ Warning: No valid images processed in {dir_name}")
            return None
        else:
            print(f"  ✓ Successfully processed {valid_images}/{len(files)} images")

        # Apply logarithmic scaling if requested
        if self.log_scale:
            frequency_map = np.log1p(frequency_map)

        # Remove vertical noise if enabled (works with float32)
        if self.remove_vertical_noise:
            frequency_map = self._remove_vertical_noise(frequency_map)

        # Create binary map for object detection
        _, binary_map = cv2.threshold(
            (frequency_map * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Filter out distant and high-frequency noise
        filtered_mask = self._filter_distant_objects(binary_map, frequency_map)

        # Apply mask to original frequency map
        final_map = frequency_map * (filtered_mask / 255.0)  # Preserve float32

        # Generate colormap (0-255 for saving)
        colormap = cv2.applyColorMap((final_map * 255).astype(np.uint8), self.colormap)
        output_filename = f"{dir_name}_colormap.png"
        output_path = os.path.join(self.output_root, output_filename)
        cv2.imwrite(output_path, colormap)
        print(f"  ✓ Created colormap: {output_path}")

        # Generate plot with normalized 0-1 values
        self._plot_frequency_map(final_map, dir_name, output_path)
        return output_path

    def _plot_frequency_map(self, frequency_map: np.ndarray, dir_name: str, output_path: str):
        """Plot the frequency map with proper 0-1 normalization"""
        plt.figure(figsize=(10, 8))
        
        # Ensure data is in 0-1 range
        normalized_map = cv2.normalize(frequency_map, None, 0, 1, cv2.NORM_MINMAX)
        
        # Plot with interpolated colors
        plot = plt.imshow(normalized_map, cmap='viridis', origin='upper', vmin=0, vmax=1, interpolation='bilinear')
        
        plt.title(f"Frequency Map: {dir_name}", fontsize=16, pad=20)
        plt.xlabel("X Pixel", fontsize=14)
        plt.ylabel("Y Pixel", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        cbar = plt.colorbar(plot, ticks=np.linspace(0, 1, 6))
        cbar.set_label("Normalized Frequency (0-1)", fontsize=14)
        
        plot_filename = f"{dir_name}_frequency_plot.png"
        plot_path = os.path.join(self.output_root, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  ✓ Saved frequency plot: {plot_path}")

    def _print_progress(self, current: int, total: int, prefix: str = ""):
        """Print a progress message with timestamp"""
        progress = f"{current}/{total}" if total else f"{current}"
        elapsed = datetime.now() - self.start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {prefix}Processing: {progress} directories | Elapsed: {elapsed}")

    def process_all(self):
        """Process all directories recursively"""
        self.start_time = datetime.now()
        
        # First pass to count total directories
        print("\n[Initializing] Scanning directory structure...")
        self.total_dirs = sum(
            len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]) > 0
            for root, _, files in os.walk(self.subtracted_root)
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {self.total_dirs} directories with images\n")

        # Second pass to process
        for root, _, files in os.walk(self.subtracted_root):
            image_files = [
                f for f in files 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
            ]
            
            if image_files:
                self.processed_dirs += 1
                dir_name = Path(root).name
                self._print_progress(self.processed_dirs, self.total_dirs, f"({dir_name}) ")
                self._process_directory(root, image_files)

        # Final completion message
        total_time = datetime.now() - self.start_time
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing completed!")
        print(f"Total directories processed: {self.processed_dirs}")
        print(f"Total processing time: {total_time}")

# Example usage
if __name__ == "__main__":
    processor = ImageProcessor(
        subtracted_root="C:/Users/garci/Desktop/Test1/output_images/grayscale/",
        output_root="C:/Users/garci/Documents/GitHub/Spray-concentration-map\output_images",
        adaptive_threshold=True,
        gaussian_sigma=1.0,
        colormap=cv2.COLORMAP_VIRIDIS,
        log_scale=True,
        remove_vertical_noise=True,
        median_kernel_size=3,
        morph_kernel_size=5,
        max_distance=50,
        min_object_area=100,
        noise_threshold=0.8
    )
    processor.process_all()