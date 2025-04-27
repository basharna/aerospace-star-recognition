import numpy as np
import itertools
from scipy.spatial import cKDTree
import cv2
import random

class TriangleMatcher:
    def __init__(self, tolerance=1e-2, angle_tolerance=1.0, match_distance=10.0, min_match_threshold=0.25, min_match_count=3, max_tries=500):
        """
        Initialize the triangle matcher.
        
        Args:
            tolerance (float): Tolerance for side length ratio matching
            angle_tolerance (float): Tolerance for angle matching in degrees
            match_distance (float): Maximum distance for point matching after transform
            min_match_count (int): Minimum number of stars that must match to consider a valid transformation
            max_tries (int): Maximum number of random triangles to try
        """
        self.tolerance = tolerance
        self.angle_tolerance = angle_tolerance
        self.match_distance = match_distance
        self.min_match_count = min_match_count
        self.max_tries = max_tries  # Maximum number of random triangles to try
        self.min_match_threshold = min_match_threshold
    def _compute_image_bounds(self, points):
        """Compute the bounding box of points."""
        if len(points) == 0:
            return 0, 0, 0, 0
        points = np.array(points)
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        return min_x, min_y, max_x, max_y
        
    def _compute_angles(self, p1, p2, p3):
        """Compute angles of a triangle in degrees."""
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)
        v3 = np.array(p3) - np.array(p2)
        
        def angle_between(v1, v2):
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.degrees(np.arccos(cos_angle))
        
        angles = [
            angle_between(v1, v2),
            angle_between(-v1, v3),
            angle_between(-v2, -v3)
        ]
        return sorted(angles)

    def _is_valid_triangle(self, p1, p2, p3, points, min_angle=15.0, min_side_ratio=0.2):
        """Check if triangle is valid for matching."""
        # Get image bounds
        min_x, min_y, max_x, max_y = self._compute_image_bounds(points)
        image_diagonal = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
        min_side_length = image_diagonal * 0.05  # 5% of image diagonal
        max_side_length = image_diagonal * 0.8   # 80% of image diagonal
        
        # Get side lengths
        sides = sorted([
            np.linalg.norm(np.array(p1) - np.array(p2)),
            np.linalg.norm(np.array(p2) - np.array(p3)),
            np.linalg.norm(np.array(p1) - np.array(p3))
        ])
        
        # Check side lengths relative to image size
        if sides[0] < min_side_length or sides[2] > max_side_length:
            return False
            
        # Check minimum side ratio
        if sides[0] / sides[2] < min_side_ratio:
            return False
        
        # Check angles
        angles = self._compute_angles(p1, p2, p3)
        if angles[0] < min_angle:  # Smallest angle too small
            return False
            
        return True

    def _triangle_signature(self, p1, p2, p3, points):
        """Compute signature of a triangle."""
        if not self._is_valid_triangle(p1, p2, p3, points):
            return None
            
        # Get side lengths
        sides = sorted([
            np.linalg.norm(np.array(p1) - np.array(p2)),
            np.linalg.norm(np.array(p2) - np.array(p3)),
            np.linalg.norm(np.array(p1) - np.array(p3))
        ])
        
        # Normalize by longest side
        ratios = (sides[0]/sides[2], sides[1]/sides[2])
        angles = self._compute_angles(p1, p2, p3)
        
        # Round to handle numerical precision
        ratios = tuple(round(r / self.tolerance) * self.tolerance for r in ratios)
        angles = tuple(round(a / self.angle_tolerance) * self.angle_tolerance for a in angles)
        
        return (ratios, angles)

    def _compute_transform(self, src_pts, dst_pts):
        """Compute transformation matrix between point sets."""
        # Convert points to the right format
        src = src_pts.astype(np.float32)
        dst = dst_pts.astype(np.float32)
        
        # Calculate centroid and scale for both point sets
        src_centroid = np.mean(src, axis=0)
        dst_centroid = np.mean(dst, axis=0)
        
        # Center the points
        src_centered = src - src_centroid
        dst_centered = dst - dst_centroid
        
        # Calculate scale
        src_scale = np.sqrt(np.mean(np.sum(src_centered**2, axis=1)))
        dst_scale = np.sqrt(np.mean(np.sum(dst_centered**2, axis=1)))
        
        if src_scale < 1e-6 or dst_scale < 1e-6:
            return None
            
        # Normalize points
        src_normalized = src_centered / src_scale
        dst_normalized = dst_centered / dst_scale
        
        # Calculate rotation matrix using SVD
        H = src_normalized.T @ dst_normalized
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation matrix (handle reflection case)
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        
        # Calculate scale as median ratio of distances from centroid
        src_dists = np.linalg.norm(src_centered, axis=1)
        dst_dists = np.linalg.norm(dst_centered, axis=1)
        scale_ratios = dst_dists / (src_dists + 1e-10)
        s = np.median(scale_ratios)
        
        # Build affine transformation matrix
        M = np.zeros((2, 3), dtype=np.float32)
        M[:2, :2] = s * R
        M[:, 2] = dst_centroid - s * R @ src_centroid
        
        return M

    def _try_transform(self, stars1, stars2, src_indices, dst_indices):
        """Try to find transformation between two sets of points."""
        src_pts = np.array([stars1[i] for i in src_indices])
        dst_pts = np.array([stars2[i] for i in dst_indices])
        
        # Compute transformation
        M = self._compute_transform(src_pts, dst_pts)
        if M is None:
            return None, 0
            
        # Apply transformation to all points
        ones = np.ones((len(stars1), 1))
        points_homog = np.hstack([stars1, ones])
        transformed = points_homog @ M.T
        
        # Count matches using KD-tree
        tree = cKDTree(stars2)
        dists, _ = tree.query(transformed, distance_upper_bound=self.match_distance)
        matches = np.sum(dists != np.inf)
        
        return transformed, matches

    def _build_triangle_db(self, stars):
        """Build a hash table of triangle signatures from the second image."""
        triangle_db = {}
        for i, j, k in itertools.combinations(range(len(stars)), 3):
            sig = self._triangle_signature(stars[i], stars[j], stars[k], stars)
            if sig:
                triangle_db.setdefault(sig, []).append((i, j, k))
        return triangle_db

    def find_match(self, stars1, stars2):
        """Find best matching triangle pattern between two star sets."""
        print(f"\nSearching for matches between {len(stars1)} and {len(stars2)} stars")
        
        # Calculate minimum required matches (30% of stars1)
        min_required_matches = int(self.min_match_threshold * len(stars1))
        self.min_match_count = max(self.min_match_count, min_required_matches)
        
        # Build hash table of triangles from image 2
        print("Building triangle database from reference image...")
        triangle_db = self._build_triangle_db(stars2)
        print(f"Created database with {len(triangle_db)} unique triangle patterns")
        print(f"Requiring at least {self.min_match_count} matches ({(self.min_match_count/len(stars1)*100):.1f}% of stars)")
        
        best_match = None
        best_count = self.min_match_count - 1
        best_tri1 = best_tri2 = None
        
        # Get all possible triangle vertex combinations from image 1
        all_triangles = list(itertools.combinations(range(len(stars1)), 3))
        random.shuffle(all_triangles)  # Randomize the order
        
        # Try random triangles from image 1
        num_tries = min(self.max_tries, len(all_triangles))
        print(f"Will try {num_tries} random triangles from image 1")
        
        for try_num, (i, j, k) in enumerate(all_triangles[:num_tries]):
            # print(f"Try {try_num + 1}/{num_tries}...")
            
            # Get triangle signature
            tri1_pts = [stars1[i], stars1[j], stars1[k]]
            sig = self._triangle_signature(*tri1_pts, stars1)
            if sig is None:
                continue
            
            # Check all matching triangles in the database
            if sig in triangle_db:
                for i2, j2, k2 in triangle_db[sig]:
                    transformed, count = self._try_transform(
                        stars1, stars2,
                        [i, j, k],
                        [i2, j2, k2]
                    )
                    
                    if transformed is not None and count > best_count:
                        print(f"New best match! {count} stars aligned ({(count/len(stars1)*100):.1f}%)")
                        best_match = transformed
                        best_count = count
                        best_tri1 = np.array(tri1_pts)
                        best_tri2 = np.array([stars2[i2], stars2[j2], stars2[k2]])
        
        if best_match is not None and best_count >= self.min_match_count:
            print(f"\nBest match found: {best_count} stars aligned ({(best_count/len(stars1)*100):.1f}%)")
            return best_match, best_count, best_tri1, best_tri2
        
        print(f"\nNo good matches found meeting the {self.min_match_threshold*100}% threshold")
        return None, 0, None, None

    def _build_kdtree(self, points):
        """Build a KD-tree for the given points."""
        return cKDTree(points) 