# infer.py
import torch
import cv2
import numpy as np
from models.pairnet import PairNet
from models.unet_refine import UNetRefine
import joblib
from data_utils import extract_sift_on_entropy, extract_patch

def load_model(use_refinement=False):
    meta = joblib.load("saved_models/meta.pkl")
    model = PairNet()
    model.load_state_dict(torch.load("saved_models/pairnet.pt"))
    model.eval()
    
    refine_model = None
    if use_refinement:
        refine_model = UNetRefine(n_channels=1, n_classes=1)
        # Load refinement model if available
        try:
            refine_model.load_state_dict(torch.load("saved_models/unet_refine.pt"))
            refine_model.eval()
        except:
            print("Refinement model not found, proceeding without refinement")
            refine_model = None
    
    return model, meta, refine_model

def create_heatmap(img_shape, suspicious_pairs, kernel_size=15):
    """Create a heatmap from suspicious pairs"""
    heatmap = np.zeros(img_shape[:2], dtype=np.float32)
    
    for kp1, kp2, prob in suspicious_pairs:
        x1, y1 = map(int, kp1.pt)
        x2, y2 = map(int, kp2.pt)
        
        # Add Gaussian at both points
        cv2.circle(heatmap, (x1, y1), kernel_size, prob, -1)
        cv2.circle(heatmap, (x2, y2), kernel_size, prob, -1)
        
        # Add line between points
        line_mask = np.zeros(img_shape[:2], dtype=np.float32)
        cv2.line(line_mask, (x1, y1), (x2, y2), prob, kernel_size)
        heatmap = np.maximum(heatmap, line_mask)
    
    # Normalize heatmap
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    return heatmap

def refine_detection(heatmap, refine_model):
    """Refine the detection using UNet"""
    if refine_model is None:
        return heatmap
    
    # Prepare input for UNet
    input_tensor = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        refined = refine_model(input_tensor)
    
    return refined.squeeze().numpy()

def predict(image_path, threshold=0.7, use_refinement=False):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None, []
    
    model, meta, refine_model = load_model(use_refinement)
    kps, descs, gray, ent = extract_sift_on_entropy(img,
        scale=meta["scale"], entropy_radius=meta["entropy_radius"])
    
    # Extract patches for all keypoints
    patches = []
    for kp in kps[:100]:  # Limit to 100 keypoints for efficiency
        patch = extract_patch(gray, kp, meta["patch_size"])
        patches.append((kp, patch))
    
    # Compare all pairs of patches
    suspicious_pairs = []
    for i in range(len(patches)):
        kp1, patch1 = patches[i]
        patch1_tensor = torch.tensor(patch1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)/255.0
        
        for j in range(i+1, len(patches)):
            kp2, patch2 = patches[j]
            patch2_tensor = torch.tensor(patch2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)/255.0
            
            with torch.no_grad():
                prob = model(patch1_tensor, patch2_tensor).item()
            
            if prob > threshold:
                suspicious_pairs.append((kp1, kp2, prob))
    
    # Create heatmap
    heatmap = create_heatmap(img.shape, suspicious_pairs)
    
    # Refine detection if requested
    if use_refinement and refine_model:
        heatmap = refine_detection(heatmap, refine_model)
    
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend with original image
    result_img = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)
    
    # Draw keypoints and lines
    for kp1, kp2, prob in suspicious_pairs:
        x1, y1 = map(int, kp1.pt)
        x2, y2 = map(int, kp2.pt)
        
        # Draw circles around keypoints
        cv2.circle(result_img, (x1, y1), 10, (0, 0, 255), 2)
        cv2.circle(result_img, (x2, y2), 10, (0, 0, 255), 2)
        
        # Draw line between matching keypoints
        cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Save and return result
    output_path = image_path.replace(".", "_result.")
    cv2.imwrite(output_path + ".png", result_img)
    
    print(f"Found {len(suspicious_pairs)} suspicious patch pairs")
    print(f"Result saved to {output_path}")
    
    return result_img, suspicious_pairs, heatmap

# Example run
if __name__ == "__main__":
    # Test on a tampered image
    predict("data/CASIA2/Tp/Tp_D_CND_M_N_ani00018_sec00096_00138.tif", use_refinement=False)