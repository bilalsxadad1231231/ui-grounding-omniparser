"""
FastAPI server for YOLO OCR detection
Accepts images and returns bounding boxes with text
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
import numpy as np
import logging
import sys
from util.utils import get_yolo_model, get_som_labeled_img
import easyocr
from paddleocr import PaddleOCR
import time
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

app = FastAPI(title="YOLO OCR Detection API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
models = {
    'yolo_model': None,
    'ocr_reader': None,
    'paddle_ocr': None,
    'models_loaded': False
}

def load_all_models():
    """Load all models once - automatically uses CUDA if available"""
    if models['models_loaded']:
        return
    
    logging.info("Loading all models...")
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        logging.info(f"🚀 CUDA detected! Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("💻 Using CPU for processing")
    
    # Load YOLO model
    models['yolo_model'] = get_yolo_model(model_path='weights/icon_detect/model.pt')
    
    # Load OCR models
    try:
        models['ocr_reader'] = easyocr.Reader(['en'], gpu=cuda_available)
        if cuda_available:
            logging.info("✅ EasyOCR loaded with GPU acceleration")
        else:
            logging.info("ℹ️ EasyOCR loaded with CPU")
    except Exception as e:
        logging.warning(f"⚠️ GPU not available for EasyOCR, using CPU: {e}")
        models['ocr_reader'] = easyocr.Reader(['en'], gpu=False)
    
    # Configure PaddleOCR
    models['paddle_ocr'] = PaddleOCR(lang='en')
    
    if cuda_available:
        logging.info("✅ PaddleOCR loaded with GPU acceleration")
    else:
        logging.info("ℹ️ PaddleOCR loaded with CPU")
    
    models['models_loaded'] = True
    logging.info("✅ All models loaded successfully!")

def fast_ocr_detection(image, use_paddleocr=True, high_quality=False):
    """OCR detection with quality options - optimized for speed"""
    # Convert to numpy array efficiently
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    w, h = image.size if hasattr(image, 'size') else image.shape[:2][::-1]
    
    if use_paddleocr and models['paddle_ocr']:
        # Use PaddleOCR
        logging.info("Using PaddleOCR")
        result = models['paddle_ocr'].ocr(image_np)[0]
        if result is None:
            return [], []
        # Adjust threshold based on quality mode
        threshold = 0.2 if high_quality else 0.3
        filtered = [(item[0], item[1][0]) for item in result if item[1][1] > threshold]
        coord = [item[0] for item in filtered]
        text = [item[1] for item in filtered]
    else:
        # Use EasyOCR
        logging.info("Using EasyOCR")
        if high_quality:
            # High quality settings
            result = models['ocr_reader'].readtext(
                image_np, 
                paragraph=False,
                text_threshold=0.3,
                width_ths=0.7,
                height_ths=0.7,
                batch_size=1,
                allowlist=None,
                blocklist='',
                detail=1,
                rotation_info=None,
                canvas_size=min(max(w, h) * 2, 4096),
                mag_ratio=1.5,
                slope_ths=0.3,
                ycenter_ths=0.7,
                y_ths=0.7,
                x_ths=1.0,
                add_margin=0.2
            )
        else:
            # Fast settings
            result = models['ocr_reader'].readtext(
                image_np, 
                paragraph=False,
                text_threshold=0.2,
                width_ths=0.8,
                height_ths=0.8,
                batch_size=1,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?@#$%^&*()_+-=[]{}|;:"<>?/~`',
                blocklist='',
                detail=1,
                rotation_info=None,
                canvas_size=2560,
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

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_all_models()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "YOLO OCR Detection API",
        "status": "running",
        "models_loaded": models['models_loaded']
    }

@app.get("/health")
async def health_check():
    """Health check with model status"""
    return {
        "status": "healthy",
        "models_loaded": models['models_loaded'],
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    use_paddleocr: bool = False,
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    imgsz: int = 640,
    high_quality: bool = False
):
    """
    Detect objects and text in an image
    
    Parameters:
    - file: Image file (jpg, png, bmp)
    - use_paddleocr: Use PaddleOCR (True) or EasyOCR (False)
    - box_threshold: Minimum confidence for detections (0.01-0.5)
    - iou_threshold: Intersection over Union threshold (0.1-0.9)
    - imgsz: YOLO input image size (320, 640, 1024)
    - high_quality: Use high quality OCR settings (slower but more accurate)
    
    Returns:
    - JSON with bounding boxes, text, and timing information
    """
    if not models['models_loaded']:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Please wait.")
    
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = image.size
        
        start_time = time.time()
        
        # OCR detection
        ocr_start = time.time()
        text, ocr_bbox = fast_ocr_detection(image, use_paddleocr, high_quality)
        ocr_time = time.time() - ocr_start
        
        # YOLO processing
        yolo_start = time.time()
        box_overlay_ratio = original_size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image, 
            models['yolo_model'], 
            BOX_TRESHOLD=box_threshold, 
            output_coord_in_ratio=True, 
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, 
            caption_model_processor=None,
            ocr_text=text,
            iou_threshold=iou_threshold, 
            imgsz=imgsz,
            use_local_semantics=False
        )
        yolo_time = time.time() - yolo_start
        
        total_time = time.time() - start_time
        
        # Convert base64 image back to PIL for processing
        result_image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        
        # Prepare response with bounding boxes
        # Format: [x1, y1, x2, y2] for each box
        ocr_boxes = []
        for i, (txt, bbox) in enumerate(zip(text, ocr_bbox)):
            ocr_boxes.append({
                "type": "text",
                "text": txt,
                "bbox": bbox,  # [x1, y1, x2, y2]
                "confidence": 1.0  # OCR doesn't always provide confidence per box
            })
        
        # Parse icon/element boxes from parsed_content_list
        icon_boxes = []
        for item in parsed_content_list:
            if item.get('type') != 'text':  # Assuming text is already in ocr_boxes
                icon_boxes.append({
                    "type": item.get('type', 'icon'),
                    "text": item.get('text', ''),
                    "bbox": item.get('bbox', []),
                    "confidence": item.get('confidence', 0.0)
                })
        
        # Combine all boxes
        all_boxes = ocr_boxes + icon_boxes
        
        return JSONResponse({
            "success": True,
            "image_size": original_size,
            "total_elements": len(all_boxes),
            "text_elements": len(ocr_boxes),
            "icon_elements": len(icon_boxes),
            "bounding_boxes": all_boxes,
            "timing": {
                "total": round(total_time, 3),
                "ocr": round(ocr_time, 3),
                "yolo": round(yolo_time, 3)
            },
            "annotated_image_base64": dino_labled_img  # Base64 encoded annotated image
        })
        
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




