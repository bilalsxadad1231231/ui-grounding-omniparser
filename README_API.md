# YOLO OCR Detection API

FastAPI server and client for YOLO OCR object detection.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the YOLO model weights at `weights/icon_detect/model.pt`

## Running the API Server

Start the FastAPI server:
```bash
python api_yolo_ocr.py
```

The server will start on `http://localhost:8000`

You can also specify a different port:
```bash
uvicorn api_yolo_ocr:app --host 0.0.0.0 --port 8000
```

### API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health check with model status
- `POST /detect` - Detect objects and text in an image

### POST /detect Parameters

- `file`: Image file (jpg, png, bmp) - **required**
- `use_paddleocr`: Use PaddleOCR (True) or EasyOCR (False) - default: False
- `box_threshold`: Minimum confidence for detections (0.01-0.5) - default: 0.05
- `iou_threshold`: Intersection over Union threshold (0.1-0.9) - default: 0.1
- `imgsz`: YOLO input image size (320, 640, 1024) - default: 640
- `high_quality`: Use high quality OCR settings - default: False

### Response Format

```json
{
  "success": true,
  "image_size": [width, height],
  "total_elements": 10,
  "text_elements": 5,
  "icon_elements": 5,
  "bounding_boxes": [
    {
      "type": "text",
      "text": "Hello World",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95
    }
  ],
  "timing": {
    "total": 1.234,
    "ocr": 0.456,
    "yolo": 0.678
  },
  "annotated_image_base64": "..."
}
```

## Using the Client Script

Process all images in a directory:

```bash
python api_client.py /path/to/images
```

### Client Options

```bash
python api_client.py <directory> [OPTIONS]

Options:
  --api-url URL              API server URL (default: http://localhost:8000)
  --output-dir DIR           Output directory (default: <input_dir>/annotated)
  --use-paddleocr            Use PaddleOCR instead of EasyOCR
  --box-threshold FLOAT      Box threshold (default: 0.05)
  --iou-threshold FLOAT      IoU threshold (default: 0.1)
  --imgsz {320,640,1024}     YOLO image size (default: 640)
  --high-quality             Use high quality OCR settings
  --no-json                  Don't save JSON results
```

### Example Usage

```bash
# Basic usage
python api_client.py ./test_images

# With custom settings
python api_client.py ./test_images \
  --use-paddleocr \
  --box-threshold 0.1 \
  --imgsz 1024 \
  --high-quality \
  --output-dir ./results

# Custom API URL
python api_client.py ./test_images --api-url http://192.168.1.100:8000
```

## Output

The client script will:
1. Process all images in the specified directory
2. Save annotated images with bounding boxes drawn
3. Save JSON results with detection data
4. Display progress and summary statistics

Output files:
- `<image_name>_annotated.png` - Image with bounding boxes drawn
- `<image_name>_results.json` - JSON file with all detection data

## Testing the API

You can test the API using curl:

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@test_image.jpg" \
  -F "use_paddleocr=false" \
  -F "box_threshold=0.05"
```

Or using Python requests:

```python
import requests

with open('test_image.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'use_paddleocr': False,
        'box_threshold': 0.05
    }
    response = requests.post('http://localhost:8000/detect', files=files, data=data)
    result = response.json()
    print(result)
```

## Notes

- The API server loads models on startup (may take a minute)
- Models are cached in memory for fast subsequent requests
- GPU is automatically used if CUDA is available
- The API returns base64-encoded annotated images for convenience




