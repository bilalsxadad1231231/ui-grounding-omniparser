import streamlit as st
import torch
from PIL import Image, ImageGrab
import base64
import io
import numpy as np
from util.utils import get_yolo_model, get_som_labeled_img
import easyocr
from paddleocr import PaddleOCR
import logging
import sys
import os
from datetime import datetime
import time

# Configure logging to show in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Page configuration
st.set_page_config(
    page_title="YOLO OCR Demo",
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
if 'clipboard_image' not in st.session_state:
    st.session_state.clipboard_image = None
if 'clipboard_saved_path' not in st.session_state:
    st.session_state.clipboard_saved_path = None

@st.cache_resource
def load_all_models():
    """Load all models once with caching - automatically uses CUDA if available"""
    with st.spinner("Loading all models (this may take a moment)..."):
        # Check if CUDA (NVIDIA GPU) is available
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            st.info(f"🚀 CUDA detected! Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.info("💻 Using CPU for processing")
        
        # Load YOLO model
        yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
        
        # Load OCR models - automatically use GPU if CUDA is available
        try:
            ocr_reader = easyocr.Reader(['en'], gpu=cuda_available)
            if cuda_available:
                st.success("✅ EasyOCR loaded with GPU acceleration")
            else:
                st.info("ℹ️ EasyOCR loaded with CPU")
        except Exception as e:
            st.warning(f"⚠️ GPU not available for EasyOCR, using CPU: {e}")
            ocr_reader = easyocr.Reader(['en'], gpu=False)
        
        # Configure PaddleOCR - simplified for compatibility
        # Note: Newer PaddleOCR versions automatically detect GPU and use default optimal settings
        paddle_ocr = PaddleOCR(
            lang='en'
        )
        
        if cuda_available:
            st.success("✅ PaddleOCR loaded with GPU acceleration")
        else:
            st.info("ℹ️ PaddleOCR loaded with CPU")
        
        return yolo_model, ocr_reader, paddle_ocr

def fast_ocr_detection(image, use_paddleocr=True, high_quality=False):
    """OCR detection with quality options - optimized for speed"""
    # Convert to numpy array efficiently
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    w, h = image.size if hasattr(image, 'size') else image.shape[:2][::-1]
    if use_paddleocr and st.session_state.paddle_ocr:
        # Use PaddleOCR
        print("🔧 Using PaddleOCR for text detection...")
        logging.info("Using PaddleOCR")
        result = st.session_state.paddle_ocr.ocr(image_np)[0]
        if result is None:
            return [], []
        # Adjust threshold based on quality mode - use list comprehension with single pass
        threshold = 0.2 if high_quality else 0.3
        filtered = [(item[0], item[1][0]) for item in result if item[1][1] > threshold]
        coord = [item[0] for item in filtered]
        text = [item[1] for item in filtered]
    else:
        print("⚡ Using EasyOCR for text detection...")
        logging.info("Using EasyOCR")
        # Adjust EasyOCR settings based on quality mode
        if high_quality:
            # High quality settings - better detection, more accurate
            result = st.session_state.ocr_reader.readtext(
                image_np, 
                paragraph=False,
                text_threshold=0.3,  # Higher threshold for better accuracy
                width_ths=0.7,       # Detect more wide text
                height_ths=0.7,      # Detect more tall text
                batch_size=1,
                allowlist=None,      # No character restriction for better detection
                blocklist='',
                detail=1,
                rotation_info=None,
                canvas_size=min(max(w, h) * 2, 4096),  # Larger canvas for high quality
                mag_ratio=1.5,       # Magnification for better detection
                slope_ths=0.3,       # Detect more rotated text
                ycenter_ths=0.7,     # Detect more off-center text
                y_ths=0.7,           # Detect more overlapping text
                x_ths=1.0,
                add_margin=0.2       # More margin for better detection
            )
        else:
            # Fast settings - optimized for speed
            result = st.session_state.ocr_reader.readtext(
                image_np, 
                paragraph=False,
                text_threshold=0.2,  # Lower threshold for speed
                width_ths=0.8,       # Skip more wide text
                height_ths=0.8,      # Skip more tall text
                batch_size=1,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?@#$%^&*()_+-=[]{}|;:"<>?/~`',
                blocklist='',
                detail=1,
                rotation_info=None,
                canvas_size=2560,    # Smaller canvas for speed
                mag_ratio=1.0,       # No magnification for speed
                slope_ths=0.1,       # Skip rotated text for speed
                ycenter_ths=0.5,     # Skip off-center text for speed
                y_ths=0.5,           # Skip overlapping text
                x_ths=1.0,
                add_margin=0.1       # Minimal margin for speed
            )
        coord = [item[0] for item in result]
        text = [item[1] for item in result]
    
    # Convert coordinates to xyxy format
    from util.utils import get_xyxy
    bb = [get_xyxy(item) for item in coord]
    
    return text, bb

def process_image(image, box_threshold, iou_threshold, use_paddleocr, imgsz, max_size, preserve_quality=True):
    """Process image with YOLO and OCR with quality preservation option and timing"""
    
    # Start total timing
    total_start_time = time.time()
    
    # Store original size (avoid unnecessary copy unless needed)
    original_size = image.size
    
    # Timing: Image preparation
    prep_start = time.time()
    
    # Only resize if preserve_quality is False or if image is extremely large
    # Use faster resampling for speed when quality is not critical
    resampling_method = Image.Resampling.LANCZOS if preserve_quality else Image.Resampling.BILINEAR
    
    if not preserve_quality and max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, resampling_method)
        st.info(f"Image resized to: {new_size} for faster processing (original: {original_size})")
    elif preserve_quality and max(image.size) > 2000:
        # Only resize very large images even in quality mode (to prevent memory issues)
        ratio = 2000 / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, resampling_method)
        st.info(f"Image resized to: {new_size} for processing (original: {original_size}) - Quality mode active")
    else:
        st.success(f"Processing at full quality: {original_size}")
    
    prep_time = time.time() - prep_start
    
    # Calculate box overlay ratio for proper scaling (use original size for better quality)
    box_overlay_ratio = original_size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }
    
    # Timing: OCR detection
    ocr_start = time.time()
    ocr_engine = "PaddleOCR" if use_paddleocr else "EasyOCR"
    quality_label = "high quality" if preserve_quality else "fast"
    with st.spinner(f"Running OCR with {ocr_engine} ({quality_label} mode)..."):
        text, ocr_bbox = fast_ocr_detection(image, use_paddleocr, preserve_quality)
    ocr_time = time.time() - ocr_start
    
    # Show which OCR engine was used
    if use_paddleocr:
        st.info("🔧 Used PaddleOCR for text detection")
    else:
        st.success("⚡ Used EasyOCR for text detection (optimized for speed)")
    
    # Timing: YOLO processing
    yolo_start = time.time()
    with st.spinner("Processing with YOLO + OCR..."):
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image, 
            st.session_state.yolo_model, 
            BOX_TRESHOLD=box_threshold, 
            output_coord_in_ratio=True, 
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, 
            caption_model_processor=None,  # Disable captioning
            ocr_text=text,
            iou_threshold=iou_threshold, 
            imgsz=imgsz,
            use_local_semantics=False  # Disable captioning
        )
    yolo_time = time.time() - yolo_start
    
    # Timing: Image conversion
    convert_start = time.time()
    result_image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    convert_time = time.time() - convert_start
    
    # Total time
    total_time = time.time() - total_start_time
    
    # Create timing dictionary
    timing_info = {
        'total': total_time,
        'preparation': prep_time,
        'ocr': ocr_time,
        'yolo': yolo_time,
        'conversion': convert_time
    }
    
    return result_image, text, parsed_content_list, timing_info

# Main UI
st.markdown('<h1 class="main-header">🔍 YOLO OCR Detection Demo</h1>', unsafe_allow_html=True)

# Sidebar for parameters
st.sidebar.header("⚙️ Parameters")

# Show device status
cuda_available = torch.cuda.is_available()
if cuda_available:
    st.sidebar.success(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    st.sidebar.caption(f"CUDA Version: {torch.version.cuda}")
else:
    st.sidebar.info("💻 Using CPU")

# Load models button
if st.sidebar.button("🚀 Load All Models", type="primary"):
    with st.spinner("Loading all models (this may take a moment)..."):
        st.session_state.yolo_model, st.session_state.ocr_reader, st.session_state.paddle_ocr = load_all_models()
        st.session_state.models_loaded = True
        st.sidebar.success("All models loaded successfully!")

# Parameters
st.sidebar.subheader("🎯 Detection Settings")
box_threshold = st.sidebar.slider("Box Threshold", 0.01, 0.5, 0.05, 0.01, help="Minimum confidence for detections")
iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 0.9, 0.1, 0.05, help="Intersection over Union threshold")
imgsz = st.sidebar.selectbox("YOLO Image Size", [320, 640, 1024], index=1, help="YOLO input image size")

st.sidebar.subheader("⚡ Quality & Speed Settings")
quality_mode = st.sidebar.selectbox("Quality Mode", ["High Quality", "Balanced", "Fast"], index=0, help="High Quality preserves original image resolution, Fast reduces size for speed")
use_paddleocr = st.sidebar.checkbox("Use PaddleOCR", value=False, help="PaddleOCR vs EasyOCR (test both for your system)")
max_size = st.sidebar.slider("Max Image Size (Fast Mode)", 800, 2400, 1600, 100, help="Maximum image dimension before resizing in Fast mode (only used in Fast mode)")

# Adjust parameters based on quality mode
preserve_quality = True
if quality_mode == "Fast":
    preserve_quality = False
    box_threshold = max(box_threshold, 0.1)  # Higher threshold = fewer detections = faster
    max_size = min(max_size, 1200)
elif quality_mode == "Balanced":
    preserve_quality = True
    # Balanced mode - process at good quality but with reasonable limits
elif quality_mode == "High Quality":
    preserve_quality = True
    box_threshold = min(box_threshold, 0.03)  # Lower threshold = more detections = better accuracy

# Speed optimization tips
st.sidebar.markdown("---")
with st.sidebar.expander("💡 Speed Optimization Tips"):
    st.markdown("""
    **For Faster Processing:**
    - Use **Fast** mode for quick results
    - Set **YOLO Image Size** to 320 or 640
    - Use **EasyOCR** (usually faster than PaddleOCR)
    - Increase **Box Threshold** to reduce detections
    - Reduce **Max Image Size** in Fast mode
    
    **For Better Quality:**
    - Use **High Quality** mode
    - Set **YOLO Image Size** to 1024
    - Lower **Box Threshold** for more detections
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    
    # Clipboard paste button
    col_paste, col_info = st.columns([1, 2])
    with col_paste:
        paste_from_clipboard = st.button("📋 Paste from Clipboard", help="Paste screenshot/image from clipboard")
    
    with col_info:
        if paste_from_clipboard:
            try:
                # Get image from clipboard
                clipboard_image = ImageGrab.grabclipboard()
                
                if clipboard_image is not None:
                    # Convert to RGB if needed
                    if clipboard_image.mode != 'RGB':
                        clipboard_image = clipboard_image.convert('RGB')
                    
                    # Save clipboard image temporarily
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    saved_path = f"screenshot_{timestamp}.png"
                    clipboard_image.save(saved_path)
                    
                    # Store in session state
                    st.session_state.clipboard_image = clipboard_image
                    st.session_state.clipboard_saved_path = saved_path
                    
                    st.success(f"✅ Screenshot pasted and saved as: {saved_path}")
                    st.image(clipboard_image, caption="Pasted from Clipboard", use_column_width=True)
                else:
                    st.warning("⚠️ No image found in clipboard. Please take a screenshot first (Windows: Win+Shift+S)")
            except Exception as e:
                st.error(f"❌ Error pasting from clipboard: {str(e)}")
                st.info("💡 Tip: Take a screenshot first (Windows: Win+Shift+S or Print Screen)")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Or choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image to process with YOLO and OCR"
    )
    
    # Process button
    if st.button("🔍 Process Image", type="primary", disabled=not st.session_state.models_loaded):
        if st.session_state.models_loaded:
            # Check for clipboard image first, then uploaded file
            if 'clipboard_image' in st.session_state and st.session_state.clipboard_image is not None:
                # Use clipboard image
                image = st.session_state.clipboard_image
                st.info(f"Processing clipboard image (saved as: {st.session_state.clipboard_saved_path})")
            elif uploaded_file is not None:
                # Load image from uploaded file
                image = Image.open(uploaded_file).convert("RGB")
            else:
                st.error("Please paste from clipboard or upload an image!")
                image = None
            
            if image is not None:
                # Process image
                result_image, ocr_text, parsed_content_list, timing_info = process_image(
                    image, box_threshold, iou_threshold, use_paddleocr, imgsz, max_size, preserve_quality
                )
                
                # Store results in session state
                st.session_state.result_image = result_image
                st.session_state.ocr_text = ocr_text
                st.session_state.parsed_content_list = parsed_content_list
                st.session_state.timing_info = timing_info
                
                st.success(f"✅ Image processed successfully in {timing_info['total']:.2f} seconds!")
        else:
            st.error("Please load models first!")

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
        
        # Timing Information
        if 'timing_info' in st.session_state and st.session_state.timing_info:
            st.subheader("⏱️ Processing Time")
            timing = st.session_state.timing_info
            
            col_time1, col_time2, col_time3, col_time4 = st.columns(4)
            
            with col_time1:
                st.metric("Total Time", f"{timing['total']:.2f}s", help="Total processing time")
            
            with col_time2:
                ocr_pct = (timing['ocr'] / timing['total'] * 100) if timing['total'] > 0 else 0
                st.metric("OCR Time", f"{timing['ocr']:.2f}s", f"{ocr_pct:.1f}%", help="OCR detection time")
            
            with col_time3:
                yolo_pct = (timing['yolo'] / timing['total'] * 100) if timing['total'] > 0 else 0
                st.metric("YOLO Time", f"{timing['yolo']:.2f}s", f"{yolo_pct:.1f}%", help="YOLO processing time")
            
            with col_time4:
                other_time = timing['preparation'] + timing['conversion']
                other_pct = (other_time / timing['total'] * 100) if timing['total'] > 0 else 0
                st.metric("Other Time", f"{other_time:.2f}s", f"{other_pct:.1f}%", help="Image preparation and conversion time")
            
            # Detailed breakdown in expander
            with st.expander("📊 Detailed Timing Breakdown"):
                st.write(f"**Image Preparation:** {timing['preparation']:.3f}s")
                st.write(f"**OCR Detection:** {timing['ocr']:.3f}s")
                st.write(f"**YOLO Processing:** {timing['yolo']:.3f}s")
                st.write(f"**Image Conversion:** {timing['conversion']:.3f}s")
                st.write(f"**Total Processing Time:** {timing['total']:.3f}s")
                
                # Performance tips
                if timing['total'] > 10:
                    st.warning("⚠️ Processing took longer than 10 seconds. Consider using 'Fast' mode for better speed.")
                elif timing['total'] > 5:
                    st.info("💡 Processing took 5-10 seconds. You can use 'Balanced' mode for a good speed/quality trade-off.")
                else:
                    st.success("⚡ Great performance! Processing completed quickly.")
        
        # Download button - save as PNG for better quality
        img_buffer = io.BytesIO()
        st.session_state.result_image.save(img_buffer, format='PNG', optimize=False)
        img_buffer.seek(0)
        
        st.download_button(
            label="💾 Download Result Image (PNG - High Quality)",
            data=img_buffer.getvalue(),
            file_name="yolo_ocr_result.png",
            mime="image/png"
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
        file_name="ocr_content.txt",
        mime="text/plain"
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🔍 YOLO OCR Detection Demo | Built with Streamlit</p>
    </div>
    """, 
    unsafe_allow_html=True
)


