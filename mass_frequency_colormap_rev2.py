import cv2
import numpy as np
import os
from pathlib import Path
import logging
from typing import List, Optional
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from datetime import datetime

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
                 max_distance: int = 50,  # in pixels
                 min_object_area: int = 100,  # in pixels
                 noise_threshold: float = 0.8,
                 axis_mode: str = "pixels",  # "mm" or "pixels"
                 mm_per_pixel_x: float = 1.0,
                 mm_per_pixel_y: float = 1.0
                ):
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
        self.axis_mode = axis_mode
        self.mm_per_pixel_x = mm_per_pixel_x
        self.mm_per_pixel_y = mm_per_pixel_y
        
        if not adaptive_threshold and threshold is None:
            raise ValueError("Either set adaptive_threshold=True or provide a threshold value")
        
        os.makedirs(self.output_root, exist_ok=True)
        self.total_dirs = 0
        self.processed_dirs = 0
        self.start_time = None

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (0, 0), self.gaussian_sigma)
        return blurred

    def _binarize_image(self, image_path: str) -> Optional[np.ndarray]:
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            preprocessed = self._preprocess_image(image)
            if self.adaptive_threshold:
                thresh_val, binarized = cv2.threshold(
                    preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                logging.debug(f"Otsu's threshold for {image_path}: {thresh_val}")
            else:
                _, binarized = cv2.threshold(
                    preprocessed, self.threshold, 255, cv2.THRESH_BINARY
                )
            return binarized
        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
            return None

    def _remove_vertical_noise(self, image: np.ndarray) -> np.ndarray:
        image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        denoised = cv2.medianBlur(image_8bit, self.median_kernel_size)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.morph_kernel_size))
        opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, vertical_kernel)
        return opened.astype(np.float32) / 255.0

    def _find_main_object(self, binary_image: np.ndarray) -> Optional[tuple]:
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
        freq_normalized = cv2.normalize(frequency_map, None, 0, 1, cv2.NORM_MINMAX)
        main_centroid, _ = self._find_main_object(binary_image)
        if main_centroid is None:
            return binary_image
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
        dir_name = Path(dir_path).name
        print(f"\n● Processing directory: {dir_name}")
        print(f"  └ Found {len(files)} images")
        
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
                frequency_map += binarized / 255.0
                valid_images += 1

        if valid_images == 0:
            print(f"  ⚠ Warning: No valid images processed in {dir_name}")
            return None
        else:
            print(f"  ✓ Successfully processed {valid_images}/{len(files)} images")

        if self.log_scale:
            frequency_map = np.log1p(frequency_map)
        if self.remove_vertical_noise:
            frequency_map = self._remove_vertical_noise(frequency_map)

        _, binary_map = cv2.threshold(
            (frequency_map * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        filtered_mask = self._filter_distant_objects(binary_map, frequency_map)
        final_map = frequency_map * (filtered_mask / 255.0)

        # Optional ratio calculation
        normalized_final = cv2.normalize(final_map, None, 0, 1, cv2.NORM_MINMAX)
        _, main_mask = self._find_main_object(binary_map)
        if main_mask is None:
            print("  ⚠ Warning: Could not determine main object for ratio calculation.")
            main_object_mask = np.ones_like(normalized_final, dtype=bool)
        else:
            main_object_mask = (main_mask > 0)
        main_object_area = np.count_nonzero(main_object_mask)
        fixed_pixels_mask = main_object_mask & (normalized_final >= 0.95)
        variable_pixels_mask = main_object_mask & (normalized_final > 0.05) & (normalized_final < 0.95)
        other_pixels_mask = main_object_mask & ~(fixed_pixels_mask | variable_pixels_mask)
        variable_pixels_mask = variable_pixels_mask | other_pixels_mask
        fixed_area_count = np.count_nonzero(fixed_pixels_mask)
        variable_area_count = np.count_nonzero(variable_pixels_mask)
        total_partition = fixed_area_count + variable_area_count
        if total_partition != main_object_area:
            print(f"  ⚠ Warning: Partitioned area ({total_partition}) != main object area ({main_object_area})")
        else:
            print(f"  ✓ Main object area partitioned correctly: {main_object_area} pixels")
        ratio = fixed_area_count / variable_area_count if variable_area_count > 0 else float('inf')
        print(f"  ✓ Fixed Area (>=0.95): {fixed_area_count}, Variable Area (rest): {variable_area_count}, Ratio: {ratio:.4f}")

        # Save fixed/variable area images (diagnostic)
        fixed_area_image = np.where(fixed_pixels_mask, normalized_final, 0)
        variable_area_image = np.where(variable_pixels_mask, normalized_final, 0)
        cv2.imwrite(os.path.join(self.output_root, f"{dir_name}_fixed_area.png"),
                    (fixed_area_image * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(self.output_root, f"{dir_name}_variable_area.png"),
                    (variable_area_image * 255).astype(np.uint8))

        # Save the colormap image
        colormap_img = cv2.applyColorMap((final_map * 255).astype(np.uint8), self.colormap)
        output_filename = f"{dir_name}_colormap.png"
        output_path = os.path.join(self.output_root, output_filename)
        cv2.imwrite(output_path, colormap_img)
        print(f"  ✓ Created colormap: {output_path}")

        # Plot frequency map (matching image size, axis in mm if requested)
        self._plot_frequency_map(final_map, dir_name)
        return output_path

    def _plot_frequency_map(self, frequency_map: np.ndarray, dir_name: str):
        """
        Plot the frequency map using the same approach as your original code,
        but switch to mm on the axis if self.axis_mode == 'mm'.
        """
        normalized_map = cv2.normalize(frequency_map, None, 0, 1, cv2.NORM_MINMAX)
        height, width = normalized_map.shape

        # Determine the displayed extent & ticks
        if self.axis_mode.lower() == "mm":
            x_max = width * self.mm_per_pixel_x
            y_max = height * self.mm_per_pixel_y
            extent = [0, x_max, y_max, 0]
            # Tick every 25 mm
            xticks = np.arange(0, x_max, 25)
            yticks = np.arange(0, y_max, 25)
            xlabel = "X (mm)"
            ylabel = "Y (mm)"
        else:
            # Pixel mode
            x_max = width
            y_max = height
            extent = [0, x_max, y_max, 0]
            # Example spacing in pixel mode
            xticks = np.arange(0, x_max, 50)
            yticks = np.arange(0, y_max, 100)
            xlabel = "X Pixel"
            ylabel = "Y Pixel"

        # Figure sizing: match image dimension + space for color bar
        dpi = 100
        image_width_inch = width / dpi
        image_height_inch = height / dpi
        colorbar_width_inch = 0.5
        total_fig_width = image_width_inch + colorbar_width_inch

        fig = plt.figure(figsize=(total_fig_width, image_height_inch), dpi=dpi)
        # Create an Axes filling the left portion (for the image)
        ax_img = fig.add_axes([0, 0, image_width_inch / total_fig_width, 1])

        im = ax_img.imshow(
            normalized_map,
            origin='upper',
            vmin=0, vmax=1,
            extent=extent,
            aspect='equal',        # keep the correct aspect ratio
            cmap='viridis'
        )

        ax_img.set_title(f"Frequency Map: {dir_name}", fontsize=14, pad=20)
        ax_img.set_xlabel(xlabel, fontsize=12)
        ax_img.set_ylabel(ylabel, fontsize=12)
        ax_img.set_xticks(xticks)
        ax_img.set_yticks(yticks)
        # If you want white dashed grid lines, set color='white':
        ax_img.grid(True, linestyle='--', linewidth=0.5, color='white', alpha=0.5)

        # Create colorbar in the remaining space on the right
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(cmap='viridis', norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])

        pos = ax_img.get_position()
        cb_pad = 0.01
        cb_width = 0.05
        ax_cb = fig.add_axes([pos.x1 + cb_pad, pos.y0, cb_width, pos.height])
        cbar = fig.colorbar(sm, cax=ax_cb)
        cbar.set_label("Normalized Frequency (0-1)", fontsize=12)

        plot_filename = f"{dir_name}_frequency_plot.png"
        plot_path = os.path.join(self.output_root, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        print(f"  ✓ Saved frequency plot: {plot_path}")

    def _print_progress(self, current: int, total: int, prefix: str = ""):
        progress = f"{current}/{total}" if total else f"{current}"
        elapsed = datetime.now() - self.start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {prefix}Processing: {progress} directories | Elapsed: {elapsed}")

    def process_all(self):
        self.start_time = datetime.now()
        print("\n[Initializing] Scanning directory structure...")
        self.total_dirs = sum(
            len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]) > 0
            for root, _, files in os.walk(self.subtracted_root)
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {self.total_dirs} directories with images\n")

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

        total_time = datetime.now() - self.start_time
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing completed!")
        print(f"Total directories processed: {self.processed_dirs}")
        print(f"Total processing time: {total_time}")


if __name__ == "__main__":
    # Ask user for axis mode (mm or pixels)
    axis_choice = input("Do you want the axis in mm or in pixels? Enter 'mm' or 'pixels': ").strip().lower()
    if axis_choice == "mm":
        try:
            mm_per_pixel_x = float(input("Enter mm per pixel for x: ").strip())
            mm_per_pixel_y = float(input("Enter mm per pixel for y: ").strip())
        except ValueError:
            print("Invalid input for mm conversion. Using default 1.0 mm per pixel.")
            mm_per_pixel_x = 1.0
            mm_per_pixel_y = 1.0
    else:
        mm_per_pixel_x = 1.0
        mm_per_pixel_y = 1.0

    processor = ImageProcessor(
        subtracted_root="C:/Users/garci/Desktop/Test1/output_images/grayscale/",
        output_root="C:/Users/garci/Documents/GitHub/Spray-concentration-map/output_images",
        adaptive_threshold=True,
        gaussian_sigma=1.0,
        colormap=cv2.COLORMAP_VIRIDIS,
        log_scale=True,
        remove_vertical_noise=True,
        median_kernel_size=3,
        morph_kernel_size=5,
        max_distance=50,
        min_object_area=100,
        noise_threshold=0.8,
        axis_mode=axis_choice,
        mm_per_pixel_x=mm_per_pixel_x,
        mm_per_pixel_y=mm_per_pixel_y
    )
    processor.process_all()
