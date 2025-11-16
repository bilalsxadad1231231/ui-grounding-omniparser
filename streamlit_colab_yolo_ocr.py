import streamlit as st
import torch
from PIL import Image
import base64
import io
import numpy as np
import os
import sys
import logging

# Add the OmniParser directory to Python path
sys.path.append('/content/OmniParser')

from util.utils import get_yolo_model, get_som_labeled_img
import easyocr
from paddleocr import PaddleOCR

# Configure logging for Colab
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Page configuration
st.set_page_config(
    page_title="YOLO OCR Demo - Colab",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .colab-info {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bbdefb;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'yolo_model' not in st.session_state:
    st.session_state.yolo_model = None
if 'ocr_reader' not in st.session_state:
    st.session_state.ocr_reader = None
if 'paddle_ocr' not in st.session_state:
    st.session_state.paddle_ocr = None

@st.cache_resource
def load_all_models():
    """Load all models once with caching - OPTIMIZED FOR COLAB"""
    with st.spinner("Loading all models for Colab (this may take a moment)..."):
        # Check if weights exist
        yolo_path = '/content/OmniParser/weights/icon_detect/model.pt'
        if not os.path.exists(yolo_path):
            st.error(f"❌ YOLO weights not found at: {yolo_path}")
            st.error("Please ensure you've cloned the repo and uploaded the weights!")
            return None, None, None
        
        # Load YOLO model
        print("🔄 Loading YOLO model...")
        yolo_model = get_yolo_model(model_path=yolo_path)
        
        # Load OCR models with Colab-optimized settings
        print("🔄 Loading EasyOCR...")
        ocr_reader = easyocr.Reader(['en'], gpu=False)  # Colab CPU optimization
        
        print("🔄 Loading PaddleOCR...")
        paddle_ocr = PaddleOCR(
            lang='en',
            use_angle_cls=False,
            use_gpu=False,  # Colab CPU optimization
            show_log=False,
            max_batch_size=256,  # Smaller batch for Colab memory
            use_dilation=False,  # Disable for speed
            det_db_thresh=0.3,   # Lower threshold for speed
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            rec_batch_num=4,     # Smaller batch for Colab
            rec_char_dict_path=None,
            rec_image_shape="3, 32, 320",
            rec_algorithm='CRNN',
            rec_model_dir=None,
            rec_image_inverse=True,
            max_text_length=25,  # Limit text length for speed
            use_space_char=True,
            drop_score=0.3,      # Lower threshold for speed
            use_mp=False,        # Disable multiprocessing for Colab stability
            total_process_num=1  # Single process for Colab
        )
        
        print("✅ All models loaded successfully!")
        return yolo_model, ocr_reader, paddle_ocr

def fast_ocr_detection(image, use_paddleocr=True):
    """Fast OCR detection with pre-loaded models - COLAB OPTIMIZED"""
    image_np = np.array(image)
    w, h = image.size
    
    if use_paddleocr and st.session_state.paddle_ocr:
        # Use PaddleOCR
        print("🔧 Using PaddleOCR for text detection...")
        logging.info("Using PaddleOCR")
        result = st.session_state.paddle_ocr.ocr(image_np, cls=False)[0]
        if result is None:
            return [], []
        coord = [item[0] for item in result if item[1][1] > 0.3]
        text = [item[1][0] for item in result if item[1][1] > 0.3]
    else:
        # Use EasyOCR with ULTRA-FAST settings for Colab
        print("⚡ Using EasyOCR for text detection...")
        logging.info("Using EasyOCR")
        result = st.session_state.ocr_reader.readtext(
            image_np, 
            paragraph=False,
            text_threshold=0.2,  # Lower threshold for speed
            width_ths=0.8,       # Skip wide text
            height_ths=0.8,      # Skip tall text
            batch_size=1,        # Process one at a time
            allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?@#$%^&*()_+-=[]{}|;:"<>?/~`',
            blocklist='',
            detail=1,
            rotation_info=None,
            canvas_size=2048,    # Smaller canvas for Colab
            mag_ratio=1.0,
            slope_ths=0.1,
            ycenter_ths=0.5,
            y_ths=0.5,
            x_ths=1.0,
            add_margin=0.1
        )
        coord = [item[0] for item in result]
        text = [item[1] for item in result]
    
    # Convert coordinates to xyxy format
    from util.utils import get_xyxy
    bb = [get_xyxy(item) for item in coord]
    
    return text, bb

def process_image(image, box_threshold, iou_threshold, use_paddleocr, imgsz, max_size):
    """Process image with YOLO and OCR - COLAB OPTIMIZED"""
    
    # AGGRESSIVE image resizing for Colab speed
    max_ocr_size = min(max_size, 600)  # Smaller size for Colab
    if max(image.size) > max_ocr_size:
        ratio = max_ocr_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        st.info(f"Image resized to: {new_size} for faster Colab processing")
    
    # Calculate box overlay ratio for proper scaling
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }
    
    # Run FAST OCR with pre-loaded models
    ocr_engine = "PaddleOCR" if use_paddleocr else "EasyOCR"
    with st.spinner(f"Running fast OCR with {ocr_engine}..."):
        text, ocr_bbox = fast_ocr_detection(image, use_paddleocr)
    
    # Show which OCR engine was used
    if use_paddleocr:
        st.info("🔧 Used PaddleOCR for text detection")
    else:
        st.success("⚡ Used EasyOCR for text detection (optimized for Colab speed)")
    
    # Process with YOLO + OCR
    with st.spinner("Processing with YOLO + OCR..."):
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image, 
            st.session_state.yolo_model, 
            BOX_TRESHOLD=box_threshold, 
            output_coord_in_ratio=True, 
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, 
            caption_model_processor=None,  # Disable captioning for speed
            ocr_text=text,
            iou_threshold=iou_threshold, 
            imgsz=imgsz,
            use_local_semantics=False  # Disable captioning
        )
    
    # Convert base64 image to PIL Image
    result_image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    
    return result_image, text, parsed_content_list

# Main UI
st.markdown('<h1 class="main-header">🔍 YOLO OCR Detection Demo - Google Colab</h1>', unsafe_allow_html=True)

# Colab info banner
st.markdown("""
<div class="colab-info">
    <h4>🚀 Google Colab Optimized</h4>
    <p>This version is optimized for Google Colab with smaller memory usage and faster processing.</p>
    <p><strong>Repo Path:</strong> <code>/content/OmniParser</code></p>
    <p><strong>Weights Path:</strong> <code>/content/OmniParser/weights/icon_detect/</code></p>
</div>
""", unsafe_allow_html=True)

# Sidebar for parameters
st.sidebar.header("⚙️ Parameters")

# Load models button
if st.sidebar.button("🚀 Load All Models", type="primary"):
    with st.spinner("Loading all models (this may take a moment)..."):
        result = load_all_models()
        if result[0] is not None:
            st.session_state.yolo_model, st.session_state.ocr_reader, st.session_state.paddle_ocr = result
            st.session_state.models_loaded = True
            st.sidebar.success("All models loaded successfully!")
        else:
            st.sidebar.error("Failed to load models. Check the console for errors.")

# Parameters
st.sidebar.subheader("🎯 Detection Settings")
box_threshold = st.sidebar.slider("Box Threshold", 0.01, 0.5, 0.05, 0.01, help="Minimum confidence for detections")
iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 0.9, 0.1, 0.05, help="Intersection over Union threshold")
imgsz = st.sidebar.selectbox("YOLO Image Size", [320, 640, 1024], index=1, help="YOLO input image size")

st.sidebar.subheader("⚡ Speed Settings")
use_paddleocr = st.sidebar.checkbox("Use PaddleOCR", value=False, help="PaddleOCR vs EasyOCR (EasyOCR recommended for Colab)")
max_size = st.sidebar.slider("Max Image Size", 400, 800, 600, 100, help="Maximum image dimension (smaller = faster on Colab)")
speed_mode = st.sidebar.selectbox("Speed Mode", ["Fast", "Balanced", "Accurate"], index=0, help="Trade-off between speed and accuracy")

# Adjust parameters based on speed mode
if speed_mode == "Fast":
    box_threshold = max(box_threshold, 0.1)
    max_size = min(max_size, 500)
elif speed_mode == "Accurate":
    box_threshold = min(box_threshold, 0.03)
    max_size = max(max_size, 700)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image to process with YOLO and OCR"
    )
    
    # Process button
    if st.button("🔍 Process Image", type="primary", disabled=not st.session_state.models_loaded):
        if uploaded_file is not None and st.session_state.models_loaded:
            # Load image
            image = Image.open(uploaded_file).convert("RGB")
            
            # Process image
            result_image, ocr_text, parsed_content_list = process_image(
                image, box_threshold, iou_threshold, use_paddleocr, imgsz, max_size
            )
            
            # Store results in session state
            st.session_state.result_image = result_image
            st.session_state.ocr_text = ocr_text
            st.session_state.parsed_content_list = parsed_content_list
            
            st.success("Image processed successfully!")
        elif not st.session_state.models_loaded:
            st.error("Please load models first!")
        else:
            st.error("Please upload an image!")

with col2:
    st.header("📊 Results")
    
    if st.session_state.models_loaded:
        st.success("✅ Models loaded and ready!")
    else:
        st.warning("⚠️ Please load models first")
    
    # Display results if available
    if 'result_image' in st.session_state:
        st.subheader("🎯 Detection Results")
        st.image(st.session_state.result_image, caption="YOLO + OCR Detection Result", use_column_width=True)
        
        # Metrics
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        
        with col_metrics1:
            st.metric("Total Elements", len(st.session_state.parsed_content_list))
        
        with col_metrics2:
            st.metric("Text Elements", len(st.session_state.ocr_text))
        
        with col_metrics3:
            icon_count = len(st.session_state.parsed_content_list) - len(st.session_state.ocr_text)
            st.metric("Icon Elements", icon_count)
        
        # Download button
        img_buffer = io.BytesIO()
        st.session_state.result_image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        st.download_button(
            label="💾 Download Result Image",
            data=img_buffer.getvalue(),
            file_name="yolo_ocr_result_colab.jpg",
            mime="image/jpeg"
        )

# OCR Text Results
if 'ocr_text' in st.session_state and st.session_state.ocr_text:
    st.header("📝 OCR Text Results")
    
    # Create expandable sections for text
    for i, text in enumerate(st.session_state.ocr_text):
        with st.expander(f"Text Box {i+1}: {text[:50]}{'...' if len(text) > 50 else ''}"):
            st.text(text)
    
    # Download OCR text
    ocr_text_content = '\n'.join([f'Text Box {i+1}: {txt}' for i, txt in enumerate(st.session_state.ocr_text)])
    st.download_button(
        label="📄 Download OCR Text",
        data=ocr_text_content,
        file_name="ocr_content_colab.txt",
        mime="text/plain"
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🔍 YOLO OCR Detection Demo - Google Colab Edition | Built with Streamlit</p>
    </div>
    """, 
    unsafe_allow_html=True
)
