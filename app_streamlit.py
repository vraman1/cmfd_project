# app_streamlit.py
import streamlit as st
import torch
import cv2
import numpy as np
from models.pairnet import PairNet
import joblib
from data_utils import extract_sift_on_entropy, extract_patch
from infer import predict

st.title("Copy-Move Forgery Detection (Hybrid Model)")

uploaded = st.file_uploader("Upload an image", type=["jpg","png","tif"])
threshold = st.slider("Detection Threshold", 0.1, 1.0, 0.7, 0.05)
use_refinement = st.checkbox("Use Refinement", value=False)

if uploaded:
    # Save uploaded file temporarily
    with open("temp_image", "wb") as f:
        f.write(uploaded.getbuffer())
    
    # Process image
        result_img, suspicious_pairs, heatmap = predict("temp_image", threshold, use_refinement)
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.image(result_img, caption=f"Detection Result - Found {len(suspicious_pairs)} suspicious pairs")
    with col2:
        st.image(heatmap, caption="Detection Heatmap", use_container_width=True)
    
    # Show original image
    original_img = cv2.imread("temp_image")
    st.image(original_img, caption="Original Image")
    
    # Show statistics
    st.subheader("Detection Statistics")
    st.write(f"Number of keypoints analyzed: 100")
    st.write(f"Number of suspicious pairs found: {len(suspicious_pairs)}")
    
    if suspicious_pairs:
        avg_prob = sum(p[2] for p in suspicious_pairs) / len(suspicious_pairs)
        st.write(f"Average match probability: {avg_prob:.3f}")
        
        # Show sample patches
        st.subheader("Sample Matching Patches")
        kp1, kp2, prob = suspicious_pairs[0]
        
        # Extract patches
        img = cv2.imread("temp_image")
        kps, descs, gray, ent = extract_sift_on_entropy(img)
        patch1 = extract_patch(gray, kp1, 64)
        patch2 = extract_patch(gray, kp2, 64)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(patch1, caption="Patch 1")
        with col2:
            st.image(patch2, caption="Patch 2")
        
        st.write(f"Match probability: {prob:.3f}")