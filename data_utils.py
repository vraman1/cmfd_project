# data_utils.py
import cv2
import numpy as np
import os
import glob
import random
from torch.utils.data import Dataset
import torch
import joblib
from tqdm import tqdm

def compute_entropy_image(gray_img, radius=3):
    """
    Custom entropy calculation without skimage dependency
    """
    # Convert to uint8 if needed
    if gray_img.dtype != np.uint8:
        gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Simple global entropy calculation
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]  # Remove zeros for log calculation
    entropy_value = -np.sum(hist * np.log2(hist))
    
    # Return constant entropy map (simplified but functional)
    return np.full_like(gray_img, entropy_value, dtype=np.float32)

def resize_image(img, scale=2):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

def extract_sift_on_entropy(img, scale=2, entropy_radius=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray_resized = resize_image(gray, scale)
    
    # Use more aggressive SIFT parameters to detect more keypoints
    sift = cv2.SIFT_create(nfeatures=2000, 
                          contrastThreshold=0.01,  # Lower = more keypoints
                          edgeThreshold=20)        # Higher = more keypoints on edges
    kps = sift.detect(gray_resized, None)
    kps, descs = sift.compute(gray_resized, kps)
    
    ent = np.zeros_like(gray_resized, dtype=np.float32)
    return kps, descs, gray_resized, ent

def extract_patch(img, kp, size=64):
    x, y = map(int, kp.pt)
    half = size // 2
    h, w = img.shape[:2]
    x1, y1 = max(0, x-half), max(0, y-half)
    x2, y2 = min(w, x+half), min(h, y+half)
    patch = img[y1:y2, x1:x2]
    patch_resized = cv2.resize(patch, (size, size))
    return patch_resized

class PairDataset(Dataset):
    def __init__(self, pairs, labels, transform=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        y = self.labels[idx]
        a = torch.tensor(a, dtype=torch.float32).unsqueeze(0)/255.0
        b = torch.tensor(b, dtype=torch.float32).unsqueeze(0)/255.0
        return a, b, torch.tensor(y, dtype=torch.float32)

def load_casia_dataset(data_path):
    """
    Load CASIA v2 dataset and return image paths with their labels
    """
    authentic_paths = glob.glob(f"{data_path}/Au/*.jpg") + glob.glob(f"{data_path}/Au/*.tif")
    tampered_paths = glob.glob(f"{data_path}/Tp/*.jpg") + glob.glob(f"{data_path}/Tp/*.tif")
    
    # Create ground truth mapping - use correct folder name with space
    gt_mapping = {}
    gt_paths = glob.glob(f"{data_path}/CASIA 2 Groundtruth/*.png")
    
    print(f"Found {len(gt_paths)} ground truth files")
    
    for gt_path in gt_paths:
        filename = os.path.basename(gt_path)
        
        # Handle the CASIA naming convention: Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png
        if filename.endswith('_gt.png'):
            # Create mappings for both .tif and .jpg extensions
            original_name_tif = filename.replace('_gt.png', '.tif')
            original_name_jpg = filename.replace('_gt.png', '.jpg')
            
            gt_mapping[original_name_tif] = gt_path
            gt_mapping[original_name_jpg] = gt_path
    
    print(f"Created {len(gt_mapping)} ground truth mappings")
    
    # Debug: show some examples
    if gt_mapping:
        print("Sample mappings:")
        for i, (img_name, mask_path) in enumerate(list(gt_mapping.items())[:3]):
            print(f"  {img_name} → {os.path.basename(mask_path)}")
    
    return authentic_paths, tampered_paths, gt_mapping

def extract_pairs_without_masks(img_paths, num_pairs_per_image=5, patch_size=64, is_tampered=False):
    """
    Extract pairs from images without needing ground truth masks
    """
    pairs = []
    labels = []
    
    for img_path in tqdm(img_paths[:min(50, len(img_paths))]):
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        kps, descs, gray, ent = extract_sift_on_entropy(img)
        
        if len(kps) < 2:
            continue
            
        # Generate positive pairs (similar patches from same image)
        if is_tampered:
            for i in range(min(num_pairs_per_image, len(kps) - 1)):
                kp1 = kps[i]
                kp2 = kps[i + 1]
                
                patch1 = extract_patch(gray, kp1, patch_size)
                patch2 = extract_patch(gray, kp2, patch_size)
                
                pairs.append((patch1, patch2))
                labels.append(1)  # Positive pair
        
        # Generate negative pairs (random patches from same image)
        for i in range(num_pairs_per_image):
            if len(kps) >= 2:
                idx1, idx2 = random.sample(range(len(kps)), 2)
                kp1 = kps[idx1]
                kp2 = kps[idx2]
                
                patch1 = extract_patch(gray, kp1, patch_size)
                patch2 = extract_patch(gray, kp2, patch_size)
                
                pairs.append((patch1, patch2))
                labels.append(0)  # Negative pair
    
    return pairs, labels

def create_dummy_pairs(num_pairs, patch_size=64):
    """
    Create dummy training pairs for testing when real data is not available
    """
    pairs = []
    labels = []
    
    for i in range(num_pairs):
        # Create random patches
        patch1 = np.random.randint(0, 255, (patch_size, patch_size), dtype=np.uint8)
        patch2 = np.random.randint(0, 255, (patch_size, patch_size), dtype=np.uint8)
        
        # 50% positive pairs (similar patches), 50% negative pairs
        if i % 2 == 0:
            # Positive pair: make second patch similar to first
            patch2 = patch1.copy() + np.random.randint(-10, 10, (patch_size, patch_size), dtype=np.int16)
            patch2 = np.clip(patch2, 0, 255).astype(np.uint8)
            label = 1
        else:
            # Negative pair: completely different patches
            label = 0
            
        pairs.append((patch1, patch2))
        labels.append(label)
    
    print(f"Created {len(pairs)} dummy pairs for testing")
    return pairs, labels

def generate_pairs_from_casia(data_path, num_pairs=1000, patch_size=64):
    """
    Generate positive and negative pairs from CASIA v2 dataset
    """
    authentic_paths, tampered_paths, gt_mapping = load_casia_dataset(data_path)
    pairs = []
    labels = []
    
    print(f"Found {len(authentic_paths)} authentic images, {len(tampered_paths)} tampered images")
    print(f"Ground truth mappings: {len(gt_mapping)}")
    
    # Show sample mapping to verify it's working
    if gt_mapping and tampered_paths:
        sample_tampered = tampered_paths[0]
        sample_name = os.path.basename(sample_tampered)
        mapped_gt = gt_mapping.get(sample_name, 'NOT FOUND')
        print(f"Sample check: {sample_name} -> {os.path.basename(mapped_gt) if mapped_gt != 'NOT FOUND' else 'NOT FOUND'}")
    
    # If no ground truth found, use alternative approach
    if len(gt_mapping) == 0:
        print("No ground truth masks found. Using alternative pair generation...")
        
        # Use tampered images for positive pairs
        print("Generating positive pairs from tampered images...")
        pos_pairs, pos_labels = extract_pairs_without_masks(tampered_paths, 5, patch_size, is_tampered=True)
        
        # Use authentic images for negative pairs  
        print("Generating negative pairs from authentic images...")
        neg_pairs, neg_labels = extract_pairs_without_masks(authentic_paths, 5, patch_size, is_tampered=False)
        
        pairs = pos_pairs + neg_pairs
        labels = pos_labels + neg_labels
        
        print(f"Generated {len(pos_pairs)} positive pairs and {len(neg_pairs)} negative pairs")
    else:
        # Original logic with masks
        print("Generating positive pairs from tampered images...")
        successful_images = 0
        for img_path in tqdm(tampered_paths[:min(100, len(tampered_paths))]):
            img_name = os.path.basename(img_path)
            gt_path = gt_mapping.get(img_name)
            
            if not gt_path or not os.path.exists(gt_path):
                continue
                
            img = cv2.imread(img_path)
            
            # FIXED MASK LOADING - Handle RGBA and various formats
            gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            if gt is None:
                continue
            
            # Handle RGBA masks (4 channels) - extract alpha channel or convert to grayscale
            if len(gt.shape) == 3 and gt.shape[2] == 4:  # RGBA image
                # Use alpha channel if it exists, otherwise convert to grayscale
                if gt[:, :, 3].max() > 0:  # Alpha channel has data
                    gt = gt[:, :, 3]  # Use alpha channel as mask
                else:
                    gt = cv2.cvtColor(gt, cv2.COLOR_BGRA2GRAY)
            elif len(gt.shape) == 3:  # RGB image
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
            
            # Ensure binary mask (0 and 255)
            _, gt = cv2.threshold(gt, 1, 255, cv2.THRESH_BINARY)
            
            # DEBUG: Print mask info for first few images
            if successful_images < 3:
                print(f"Mask: {os.path.basename(gt_path)} | Size: {gt.shape} | Non-zero: {(gt > 0).sum()} pixels")
            
            # Get SIFT keypoints
            kps, descs, gray, ent = extract_sift_on_entropy(img)
            
            # Only consider keypoints in tampered regions (where mask > 0)
            valid_kps = []
            total_kps = len(kps)
            for kp in kps:
                x, y = map(int, kp.pt)
                if y < gt.shape[0] and x < gt.shape[1] and gt[y, x] > 0:
                    valid_kps.append(kp)
            
            # DEBUG: Print keypoint info for first few images
            if successful_images < 3:
                print(f"Keypoints: {total_kps} total, {len(valid_kps)} in masked regions")
            
            # Generate positive pairs from the same tampered region
            if len(valid_kps) >= 2:
                successful_images += 1
                for i in range(min(5, len(valid_kps))):
                    if i + 1 >= len(valid_kps):
                        break
                        
                    kp1 = valid_kps[i]
                    kp2 = valid_kps[i + 1]
                    
                    patch1 = extract_patch(gray, kp1, patch_size)
                    patch2 = extract_patch(gray, kp2, patch_size)
                    
                    pairs.append((patch1, patch2))
                    labels.append(1)  # Positive pair
        
        print(f"Successfully processed {successful_images} tampered images with masks")
        
        print("Generating negative pairs from authentic images...")
        # Negative pairs: patches from authentic images (should not match)
        for img_path in tqdm(authentic_paths[:min(100, len(authentic_paths))]):
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            kps, descs, gray, ent = extract_sift_on_entropy(img)
            
            if len(kps) >= 2:
                for i in range(min(5, len(kps))):
                    if i + 1 >= len(kps):
                        break
                        
                    kp1 = kps[i]
                    kp2 = kps[i + 1]
                    
                    patch1 = extract_patch(gray, kp1, patch_size)
                    patch2 = extract_patch(gray, kp2, patch_size)
                    
                    pairs.append((patch1, patch2))
                    labels.append(0)  # Negative pair
    
    # If still no pairs, use dummy data
    if len(pairs) == 0:
        print("Warning: No pairs generated! Using dummy data...")
        pairs, labels = create_dummy_pairs(500, patch_size)
    
    print(f"Generated {len(pairs)} total pairs")
    return pairs, labels

def save_pairs(pairs, labels, save_path):
    """Save generated pairs to disk"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({"pairs": pairs, "labels": labels}, save_path)
    
def load_pairs(load_path):
    """Load generated pairs from disk"""
    if os.path.exists(load_path):
        data = joblib.load(load_path)
        return data["pairs"], data["labels"]
    return [], []