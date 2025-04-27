import matplotlib.pyplot as plt
from star_detector import StarDetector
from triangle_matcher import TriangleMatcher
import cv2
import numpy as np

# Add color constants
YELLOW = '\033[93m'
RESET = '\033[0m'

def draw_triangle(img, triangle_points, color=(0, 255, 0), thickness=2):
    """Draw a triangle and circles at its vertices."""
    # Convert points to integer coordinates
    points = triangle_points.astype(np.int32)
    
    # Draw the triangle edges
    cv2.line(img, tuple(points[0]), tuple(points[1]), color, thickness)
    cv2.line(img, tuple(points[1]), tuple(points[2]), color, thickness)
    cv2.line(img, tuple(points[2]), tuple(points[0]), color, thickness)
    
    # Draw circles at vertices
    for point in points:
        cv2.circle(img, tuple(point), 5, color, -1)  # filled circle

def main():
    # Initialize components with thresholding parameters
    detector = StarDetector(
        threshold=120,              # Brightness threshold (adjust based on image brightness)
        min_area=4,                 # Minimum star size
        max_area=2000               # Maximum star size
    )
    matcher = TriangleMatcher(
        tolerance=0.05,             # More lenient ratio matching
        angle_tolerance=1.0,        # More lenient angle matching
        match_distance=15.0,        # More lenient distance matching
        max_tries=500              # More tries for better results
    )
    
    # Image paths
    img1_path = 'Stars/IMG_3063.jpeg'
    img2_path = 'Stars/IMG_3061.jpeg'
    
    try:
        # Detect stars in both images
        stars_img1 = detector.detect_stars(img1_path)
        stars_img2 = detector.detect_stars(img2_path)
        
        if len(stars_img1) == 0 or len(stars_img2) == 0:
            print("No stars detected! Try adjusting the threshold value.")
            return
        
        print(f"Detected {len(stars_img1)} stars in image 1")
        print(f"Detected {len(stars_img2)} stars in image 2")
        
        # Validate that second image has more stars
        if len(stars_img2) <= len(stars_img1):
            print(f"\n{YELLOW}Warning: The second image should have more stars than the first image!")
            print(f"First image: {len(stars_img1)} stars")
            print(f"Second image: {len(stars_img2)} stars")
            print(f"Consider swapping the order of your input images.{RESET}\n")
            return
        
        print("Searching for best matching triangle pattern...")
        
        # Find matching triangles and transform
        transformed_stars, match_count, triangle1, triangle2 = matcher.find_match(stars_img1, stars_img2)
        
        if transformed_stars is not None:
            match_percentage = (match_count / len(stars_img1)) * 100
            print(f"Found best matching pattern!")
            print(f"Matched {match_count} stars out of {len(stars_img1)} ({match_percentage:.1f}%)")
            
            # Create visualization
            plt.figure(figsize=(12, 5))
            
            # Plot original image 1 with matching triangle
            plt.subplot(121)
            img1_vis = detector.visualize_detections(img1_path, stars_img1)
            draw_triangle(img1_vis, triangle1, color=(0, 255, 255), thickness=2)  # yellow triangle
            plt.imshow(cv2.cvtColor(img1_vis, cv2.COLOR_BGR2RGB))
            plt.title("Image 1 with Matching Triangle\n" + 'detected stars: ' + str(len(stars_img1)))
            plt.axis('off')
            
            # Plot original image 2 with matching triangle and transformed stars
            plt.subplot(122)
            img2_vis = detector.visualize_detections(img2_path, stars_img2)
            draw_triangle(img2_vis, triangle2, color=(0, 255, 255), thickness=2)  # yellow triangle
            
            # Draw transformed stars as red X markers
            for point in transformed_stars:
                point = point.astype(np.int32)
                size = 10
                thickness = 2
                color = (0, 0, 255)  # Red color in BGR
                cv2.line(img2_vis, 
                        (point[0] - size, point[1] - size),
                        (point[0] + size, point[1] + size),
                        color, thickness)
                cv2.line(img2_vis, 
                        (point[0] - size, point[1] + size),
                        (point[0] + size, point[1] - size),
                        color, thickness)
            
            # Draw lines between matched pairs
            tree = matcher._build_kdtree(stars_img2)
            dists, indices = tree.query(transformed_stars, distance_upper_bound=matcher.match_distance)
            for i, (dist, idx) in enumerate(zip(dists, indices)):
                if dist != np.inf:
                    start_point = tuple(transformed_stars[i].astype(np.int32))
                    end_point = tuple(stars_img2[idx].astype(np.int32))
                    cv2.line(img2_vis, start_point, end_point, (0, 255, 0), 1)  # Green lines with alpha=0.3
            
            plt.imshow(cv2.cvtColor(img2_vis, cv2.COLOR_BGR2RGB))
            plt.title(f"Image 2 with Matches\n{match_count} matches ({match_percentage:.1f}%)\n{len(stars_img2)} stars detected")
            plt.axis('off')
            
            # Add custom legend
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            
            legend_elements = [
                Line2D([0], [0], marker='o', color='green', markerfacecolor='green', markersize=8, label='Database Stars'),
                Line2D([0], [0], marker='x', color='red', markersize=8, label='Transformed Stars'),
                Line2D([0], [0], color='yellow', label='Matching Triangle')
            ]
            plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
            
            plt.tight_layout()
            plt.show()
        else:
            print("No matching triangle patterns found that meet the minimum match criteria.")
            print("Try adjusting the matcher parameters (tolerance, angle_tolerance, or min_match_count)")
            
    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure the image paths are correct")

if __name__ == "__main__":
    main() 