import cv2
import numpy as np

class StarDetector:
    def __init__(self, threshold=127, min_area=5, max_area=2000, min_circularity=0.5, max_aspect_ratio=1.5):
        """
        Initialize the star detector with thresholding parameters.
        
        Args:
            threshold (int): Threshold value for binary image creation (0-255)
            min_area (float): Minimum area of stars to detect
            max_area (float): Maximum area of stars to detect
            min_circularity (float): Minimum circularity ratio (0-1) for star detection
            max_aspect_ratio (float): Maximum aspect ratio for star detection
        """
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.max_aspect_ratio = max_aspect_ratio
    
    def detect_stars(self, image_path):
        """
        Detect stars in an image using thresholding and contour detection.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Array of star centroids (x, y coordinates)
        """
        # Read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Apply thresholding
        _, binary = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        star_positions = []
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.min_area <= area <= self.max_area:
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Get rotated rectangle to check aspect ratio
                    rect = cv2.minAreaRect(contour)
                    width = rect[1][0]
                    height = rect[1][1]
                    
                    # Avoid division by zero
                    if width > 0 and height > 0:
                        aspect_ratio = max(width, height) / min(width, height)
                        
                        # Filter by circularity and aspect ratio
                        if circularity > self.min_circularity and aspect_ratio <= self.max_aspect_ratio:
                            # Calculate centroid
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = M["m10"] / M["m00"]
                                cy = M["m01"] / M["m00"]
                                star_positions.append([cx, cy])
        
        return np.array(star_positions)

    def visualize_detections(self, image_path, star_positions):
        """
        Visualize detected stars on the original image.
        
        Args:
            image_path (str): Path to the original image
            star_positions (numpy.ndarray): Array of star centroids
        """
        img = cv2.imread(image_path)
        
        # Draw detected stars
        for x, y in star_positions:
            # Draw circle for each star
            cv2.circle(img, (int(x), int(y)), 10, (0, 255, 0), 2)
        
        return img 