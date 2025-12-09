"""
Client script to process images through YOLO OCR API
Takes a directory path, processes all images, displays bounding boxes, and saves results
"""
import os
import sys
import requests
import json
from PIL import Image, ImageDraw, ImageFont
import base64
import io
from pathlib import Path
import argparse
from typing import List, Dict, Tuple

class YOLOOCRClient:
    def __init__(self, api_url: str = "http://localhost:8000"):
        """
        Initialize the API client
        
        Args:
            api_url: Base URL of the FastAPI server
        """
        self.api_url = api_url.rstrip('/')
        self.detect_endpoint = f"{self.api_url}/detect"
        self.health_endpoint = f"{self.api_url}/health"
        
    def check_health(self) -> bool:
        """Check if the API server is healthy"""
        try:
            headers = {'ngrok-skip-browser-warning': 'true'}
            response = requests.get(self.health_endpoint, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"[OK] API Server is healthy")
                print(f"   Models loaded: {data.get('models_loaded', False)}")
                print(f"   CUDA available: {data.get('cuda_available', False)}")
                return True
            else:
                print(f"[ERROR] API Server returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Cannot connect to API server at {self.api_url}")
            print(f"   Error: {str(e)}")
            print(f"   Make sure the server is running: python api_yolo_ocr.py")
            return False
    
    def detect_objects(
        self, 
        image_path: str,
        use_paddleocr: bool = False,
        box_threshold: float = 0.05,
        iou_threshold: float = 0.1,
        imgsz: int = 640,
        high_quality: bool = False
    ) -> Dict:
        """
        Send image to API and get bounding boxes
        
        Args:
            image_path: Path to image file
            use_paddleocr: Use PaddleOCR (True) or EasyOCR (False)
            box_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold
            imgsz: YOLO image size
            high_quality: Use high quality OCR settings
            
        Returns:
            Dictionary with detection results
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                data = {
                    'use_paddleocr': use_paddleocr,
                    'box_threshold': box_threshold,
                    'iou_threshold': iou_threshold,
                    'imgsz': imgsz,
                    'high_quality': high_quality
                }
                
                print(f"Sending {os.path.basename(image_path)} to API...")
                headers = {'ngrok-skip-browser-warning': 'true'}
                response = requests.post(self.detect_endpoint, files=files, data=data, headers=headers, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"[OK] Received {result.get('total_elements', 0)} detections")
                    return result
                else:
                    print(f"[ERROR] API returned error: {response.status_code}")
                    print(f"   {response.text}")
                    return None
                    
        except Exception as e:
            print(f"[ERROR] Error processing {image_path}: {str(e)}")
            return None
    
    def draw_boxes_on_image(
        self, 
        image_path: str, 
        bounding_boxes: List[Dict],
        output_path: str = None,
        use_api_image: bool = True,
        annotated_image_base64: str = None
    ) -> Image.Image:
        """
        Draw bounding boxes on image
        
        Args:
            image_path: Path to original image
            bounding_boxes: List of bounding box dictionaries
            output_path: Path to save annotated image (optional)
            use_api_image: Use the annotated image from API if available
            annotated_image_base64: Base64 encoded annotated image from API
            
        Returns:
            PIL Image with bounding boxes drawn
        """
        # If API provided annotated image, use it
        if use_api_image and annotated_image_base64:
            try:
                image_bytes = base64.b64decode(annotated_image_base64)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                if output_path:
                    image.save(output_path)
                    print(f"[SAVED] Annotated image saved to {output_path}")
                return image
            except Exception as e:
                print(f"[WARNING] Could not use API annotated image, drawing manually: {e}")
        
        # Otherwise, draw boxes manually
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except:
                font = ImageFont.load_default()
        
        # Color mapping for different types
        colors = {
            'text': (0, 255, 0),      # Green
            'icon': (255, 0, 0),      # Red
            'button': (0, 0, 255),     # Blue
            'default': (255, 255, 0)   # Yellow
        }
        
        for box in bounding_boxes:
            bbox = box.get('bbox', [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                box_type = box.get('type', 'default')
                text = box.get('text', '')
                confidence = box.get('confidence', 0.0)
                
                # Get color based on type
                color = colors.get(box_type, colors['default'])
                
                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # Draw label
                label = f"{box_type}: {text[:20]}" if text else box_type
                if confidence > 0:
                    label += f" ({confidence:.2f})"
                
                # Draw text background
                bbox_text = draw.textbbox((x1, y1 - 20), label, font=font)
                draw.rectangle(bbox_text, fill=color)
                draw.text((x1, y1 - 20), label, fill=(0, 0, 0), font=font)
        
        if output_path:
            image.save(output_path)
            print(f"[SAVED] Annotated image saved to {output_path}")
        
        return image
    
    def process_directory(
        self,
        directory_path: str,
        output_dir: str = None,
        use_paddleocr: bool = False,
        box_threshold: float = 0.05,
        iou_threshold: float = 0.1,
        imgsz: int = 640,
        high_quality: bool = False,
        save_json: bool = True
    ):
        """
        Process all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            output_dir: Directory to save annotated images (default: directory_path/annotated)
            use_paddleocr: Use PaddleOCR
            box_threshold: Box threshold
            iou_threshold: IoU threshold
            imgsz: YOLO image size
            high_quality: High quality mode
            save_json: Save JSON results
        """
        # Validate directory
        if not os.path.isdir(directory_path):
            print(f"[ERROR] Directory not found: {directory_path}")
            return
        
        # Check API health
        if not self.check_health():
            return
        
        # Setup output directory
        if output_dir is None:
            output_dir = os.path.join(directory_path, "annotated")
        os.makedirs(output_dir, exist_ok=True)
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Find all images
        image_files = []
        for file in os.listdir(directory_path):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(directory_path, file))
        
        if not image_files:
            print(f"[ERROR] No image files found in {directory_path}")
            return
        
        print(f"\nFound {len(image_files)} image(s) to process")
        print(f"Output directory: {output_dir}\n")
        
        # Process each image
        results_summary = []
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
            
            # Detect objects
            result = self.detect_objects(
                image_path,
                use_paddleocr=use_paddleocr,
                box_threshold=box_threshold,
                iou_threshold=iou_threshold,
                imgsz=imgsz,
                high_quality=high_quality
            )
            
            if result and result.get('success'):
                # Prepare output paths
                base_name = Path(image_path).stem
                annotated_path = os.path.join(output_dir, f"{base_name}_annotated.png")
                json_path = os.path.join(output_dir, f"{base_name}_results.json")
                
                # Draw boxes and save
                annotated_image = self.draw_boxes_on_image(
                    image_path,
                    result.get('bounding_boxes', []),
                    output_path=annotated_path,
                    use_api_image=True,
                    annotated_image_base64=result.get('annotated_image_base64')
                )
                
                # Save JSON results
                if save_json:
                    with open(json_path, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"[SAVED] JSON results saved to {json_path}")
                
                # Print summary
                timing = result.get('timing', {})
                print(f"   Total: {timing.get('total', 0):.2f}s | "
                      f"OCR: {timing.get('ocr', 0):.2f}s | "
                      f"YOLO: {timing.get('yolo', 0):.2f}s")
                print(f"   Detections: {result.get('total_elements', 0)} total | "
                      f"{result.get('text_elements', 0)} text | "
                      f"{result.get('icon_elements', 0)} icons")
                
                results_summary.append({
                    'image': os.path.basename(image_path),
                    'total_elements': result.get('total_elements', 0),
                    'text_elements': result.get('text_elements', 0),
                    'icon_elements': result.get('icon_elements', 0),
                    'total_time': timing.get('total', 0)
                })
            else:
                print(f"   [ERROR] Failed to process image")
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"[OK] Processing complete!")
        print(f"   Processed: {len(results_summary)}/{len(image_files)} images")
        if results_summary:
            avg_time = sum(r['total_time'] for r in results_summary) / len(results_summary)
            total_elements = sum(r['total_elements'] for r in results_summary)
            print(f"   Average time: {avg_time:.2f}s per image")
            print(f"   Total detections: {total_elements}")
        print(f"   Results saved to: {output_dir}")
        print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Process images through YOLO OCR API")
    parser.add_argument(
        "input_path",
        type=str,
        help="Directory path or single image file path to process"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="API server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for annotated images (default: <input_dir>/annotated)"
    )
    parser.add_argument(
        "--use-paddleocr",
        action="store_true",
        help="Use PaddleOCR instead of EasyOCR"
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.05,
        help="Box threshold (default: 0.05)"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.1,
        help="IoU threshold (default: 0.1)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        choices=[320, 640, 1024],
        help="YOLO image size (default: 640)"
    )
    parser.add_argument(
        "--high-quality",
        action="store_true",
        help="Use high quality OCR settings"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Don't save JSON results"
    )
    
    args = parser.parse_args()
    
    # Create client
    client = YOLOOCRClient(api_url=args.api_url)
    
    # Check if input is a file or directory
    if os.path.isfile(args.input_path):
        # Process single image
        print(f"Processing single image: {args.input_path}")
        
        # Check API health
        if not client.check_health():
            return
        
        # Setup output directory
        if args.output_dir is None:
            output_dir = os.path.dirname(args.input_path) or "."
        else:
            output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Detect objects
        result = client.detect_objects(
            args.input_path,
            use_paddleocr=args.use_paddleocr,
            box_threshold=args.box_threshold,
            iou_threshold=args.iou_threshold,
            imgsz=args.imgsz,
            high_quality=args.high_quality
        )
        
        if result and result.get('success'):
            # Prepare output paths
            base_name = Path(args.input_path).stem
            annotated_path = os.path.join(output_dir, f"{base_name}_annotated.png")
            json_path = os.path.join(output_dir, f"{base_name}_results.json")
            
            # Draw boxes and save
            annotated_image = client.draw_boxes_on_image(
                args.input_path,
                result.get('bounding_boxes', []),
                output_path=annotated_path,
                use_api_image=True,
                annotated_image_base64=result.get('annotated_image_base64')
            )
            
            # Save JSON results
            if not args.no_json:
                with open(json_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"[SAVED] JSON results saved to {json_path}")
            
            # Print summary
            timing = result.get('timing', {})
            print(f"\n{'='*60}")
            print(f"[OK] Processing complete!")
            print(f"   Total: {timing.get('total', 0):.2f}s | "
                  f"OCR: {timing.get('ocr', 0):.2f}s | "
                  f"YOLO: {timing.get('yolo', 0):.2f}s")
            print(f"   Detections: {result.get('total_elements', 0)} total | "
                  f"{result.get('text_elements', 0)} text | "
                  f"{result.get('icon_elements', 0)} icons")
            print(f"   Annotated image saved to: {annotated_path}")
            print(f"{'='*60}\n")
        else:
            print(f"[ERROR] Failed to process image")
    else:
        # Process directory
        client.process_directory(
            directory_path=args.input_path,
            output_dir=args.output_dir,
            use_paddleocr=args.use_paddleocr,
            box_threshold=args.box_threshold,
            iou_threshold=args.iou_threshold,
            imgsz=args.imgsz,
            high_quality=args.high_quality,
            save_json=not args.no_json
        )

if __name__ == "__main__":
    main()




