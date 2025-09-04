#!/usr/bin/env python3
"""
Flask Image Annotation App with Auto-Annotation
===============================================
Interactive tool for annotating aerial images with manual and automatic annotation.
"""

import os
import json
import csv
import uuid
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoAnnotator:
    """Automatic annotation of aerial images using computer vision techniques"""
    
    def __init__(self):
        # Color ranges for different feature detection (in HSV)
        self.color_ranges = {
            'vegetation': {
                'lower': np.array([35, 40, 40]),
                'upper': np.array([85, 255, 255])
            },
            'water': {
                'lower': np.array([90, 50, 50]),
                'upper': np.array([120, 255, 255])
            },
            'building': {
                'lower': np.array([0, 0, 100]),
                'upper': np.array([180, 50, 255])
            },
            'road': {
                'lower': np.array([0, 0, 100]),
                'upper': np.array([180, 30, 200])
            }
        }
        
        # Annotation colors for different features
        self.annotation_colors = {
            'building': '#ff0000',  # Red
            'road': '#0000ff',      # Blue
            'vegetation': '#00ff00', # Green
            'water': '#00ffff'      # Cyan
        }
        
    def detect_features(self, image_path: str) -> List[Dict]:
        """
        Detect features in an aerial image using computer vision techniques
        Returns a list of annotation objects
        """
        annotations = []
        
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return annotations
                
            # Convert to HSV color space for better color segmentation
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            height, width = img.shape[:2]
            
            # Detect different features
            annotations.extend(self.detect_buildings(img, hsv, width, height))
            annotations.extend(self.detect_roads(img, hsv, width, height))
            annotations.extend(self.detect_vegetation(img, hsv, width, height))
            annotations.extend(self.detect_water(img, hsv, width, height))
            
            logger.info(f"Detected {len(annotations)} features in {image_path}")
            
        except Exception as e:
            logger.error(f"Error detecting features in {image_path}: {e}")
            
        return annotations
    
    def detect_buildings(self, img, hsv, width, height) -> List[Dict]:
        """Detect buildings using edge detection and contour analysis"""
        annotations = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter by area (too small or too large are likely not buildings)
                area = cv2.contourArea(contour)
                if area < 500 or area > 10000:
                    continue
                
                # Approximate the contour to a polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Buildings typically have 4 or more sides
                if len(approx) >= 4:
                    # Convert coordinates to relative values (0-1)
                    coordinates = []
                    for point in approx:
                        x = point[0][0] / width
                        y = point[0][1] / height
                        coordinates.append([x, y])
                    
                    # Create annotation
                    annotation = {
                        'type': 'polygon',
                        'label': 'building',
                        'coordinates': coordinates,
                        'color': self.annotation_colors['building'],
                        'opacity': 70,
                        'confidence': min(0.9, area / 5000)  # Simple confidence metric
                    }
                    annotations.append(annotation)
                    
        except Exception as e:
            logger.error(f"Error detecting buildings: {e}")
            
        return annotations
    
    def detect_roads(self, img, hsv, width, height) -> List[Dict]:
        """Detect roads using line detection and color thresholding"""
        annotations = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Line detection using Hough Transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Convert to relative coordinates
                    x1_rel = x1 / width
                    y1_rel = y1 / height
                    x2_rel = x2 / width
                    y2_rel = y2 / height
                    
                    # Create annotation
                    annotation = {
                        'type': 'line',
                        'label': 'road',
                        'coordinates': [[x1_rel, y1_rel], [x2_rel, y2_rel]],
                        'color': self.annotation_colors['road'],
                        'opacity': 80,
                        'confidence': 0.7
                    }
                    annotations.append(annotation)
                    
        except Exception as e:
            logger.error(f"Error detecting roads: {e}")
            
        return annotations
    
    def detect_vegetation(self, img, hsv, width, height) -> List[Dict]:
        """Detect vegetation using color thresholding in HSV space"""
        annotations = []
        
        try:
            # Create mask for vegetation (green colors in HSV)
            mask = cv2.inRange(hsv, 
                             self.color_ranges['vegetation']['lower'], 
                             self.color_ranges['vegetation']['upper'])
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1000:  # Filter small areas
                    continue
                
                # Create a bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Convert to relative coordinates
                x_rel = x / width
                y_rel = y / height
                w_rel = w / width
                h_rel = h / height
                
                # Create annotation
                annotation = {
                    'type': 'rectangle',
                    'label': 'vegetation',
                    'coordinates': [[x_rel, y_rel], [x_rel + w_rel, y_rel + h_rel]],
                    'color': self.annotation_colors['vegetation'],
                    'opacity': 60,
                    'confidence': min(0.8, area / 10000)
                }
                annotations.append(annotation)
                
        except Exception as e:
            logger.error(f"Error detecting vegetation: {e}")
            
        return annotations
    
    def detect_water(self, img, hsv, width, height) -> List[Dict]:
        """Detect water bodies using color thresholding in HSV space"""
        annotations = []
        
        try:
            # Create mask for water (blue colors in HSV)
            mask = cv2.inRange(hsv, 
                             self.color_ranges['water']['lower'], 
                             self.color_ranges['water']['upper'])
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 2000:  # Filter small areas
                    continue
                
                # Approximate the contour to a polygon
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Convert coordinates to relative values (0-1)
                coordinates = []
                for point in approx:
                    x = point[0][0] / width
                    y = point[0][1] / height
                    coordinates.append([x, y])
                
                # Create annotation
                annotation = {
                    'type': 'polygon',
                    'label': 'water',
                    'coordinates': coordinates,
                    'color': self.annotation_colors['water'],
                    'opacity': 50,
                    'confidence': min(0.85, area / 15000)
                }
                annotations.append(annotation)
                
        except Exception as e:
            logger.error(f"Error detecting water: {e}")
            
        return annotations


class ImageAnnotationApp:
    """Main application class for image annotation functionality"""
    
    def __init__(self, app: Flask):
        self.app = app
        self.setup_directories()
        self.annotations = {}  # Store annotations in memory
        self.current_session = str(uuid.uuid4())
        self.auto_annotator = AutoAnnotator()
        
    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.app.config['UPLOAD_FOLDER'],
            self.app.config['AERIAL_IMAGES_FOLDER'],
            self.app.config['ANNOTATIONS_FOLDER'],
            self.app.config['THUMBNAILS_FOLDER']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ready: {directory}")
    
    def get_supported_images(self, folder_path: str) -> List[str]:
        """Get list of supported image files from a directory"""
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = []
        
        folder = Path(folder_path)
        if not folder.exists():
            logger.warning(f"Folder does not exist: {folder_path}")
            return image_files
        
        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                image_files.append(file_path.name)
        
        logger.info(f"Found {len(image_files)} supported images in {folder_path}")
        return sorted(image_files)
    
    def create_thumbnail(self, image_path: str, size: Tuple[int, int] = (200, 200)) -> str:
        """Create thumbnail for an image"""
        try:
            input_path = Path(image_path)
            thumbnail_name = f"thumb_{input_path.stem}.jpg"
            thumbnail_path = Path(self.app.config['THUMBNAILS_FOLDER']) / thumbnail_name
            
            # Skip if thumbnail already exists
            if thumbnail_path.exists():
                return thumbnail_name
            
            # Create thumbnail
            with Image.open(input_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Create thumbnail maintaining aspect ratio
                img.thumbnail(size, Image.Resampling.LANCZOS)
                img.save(thumbnail_path, 'JPEG', quality=85, optimize=True)
                
            logger.info(f"Created thumbnail: {thumbnail_name}")
            return thumbnail_name
            
        except Exception as e:
            logger.error(f"Error creating thumbnail for {image_path}: {e}")
            return ""
    
    def get_image_info(self, image_filename: str) -> Dict:
        """Get detailed information about an image"""
        image_path = Path(self.app.config['AERIAL_IMAGES_FOLDER']) / image_filename
        
        if not image_path.exists():
            return {}
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                file_size = image_path.stat().st_size
                
            # Create thumbnail
            thumbnail_name = self.create_thumbnail(str(image_path))
            
            # Get existing annotations
            annotations = self.annotations.get(image_filename, [])
            
            return {
                'filename': image_filename,
                'width': width,
                'height': height,
                'file_size': file_size,
                'thumbnail': thumbnail_name,
                'annotation_count': len(annotations),
                'last_modified': datetime.fromtimestamp(image_path.stat().st_mtime).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting image info for {image_filename}: {e}")
            return {}
    
    def save_annotation(self, image_filename: str, annotation_data: Dict) -> bool:
        """Save an annotation for a specific image"""
        try:
            # Validate annotation data
            required_fields = ['type', 'coordinates', 'label']
            if not all(field in annotation_data for field in required_fields):
                logger.error(f"Missing required fields in annotation data: {annotation_data}")
                return False
            
            # Add metadata
            annotation = {
                'id': str(uuid.uuid4()),
                'type': annotation_data['type'],
                'label': annotation_data.get('label', ''),
                'coordinates': annotation_data['coordinates'],
                'color': annotation_data.get('color', '#ff0000'),
                'opacity': annotation_data.get('opacity', 50),
                'created_at': datetime.now().isoformat(),
                'session_id': self.current_session
            }
            
            # Initialize image annotations if not exists
            if image_filename not in self.annotations:
                self.annotations[image_filename] = []
            
            # Add annotation
            self.annotations[image_filename].append(annotation)
            
            logger.info(f"Saved annotation for {image_filename}: {annotation['type']} - {annotation['label']}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving annotation: {e}")
            return False
    
    def delete_annotation(self, image_filename: str, annotation_id: str) -> bool:
        """Delete a specific annotation"""
        try:
            if image_filename not in self.annotations:
                return False
            
            self.annotations[image_filename] = [
                ann for ann in self.annotations[image_filename] 
                if ann['id'] != annotation_id
            ]
            
            logger.info(f"Deleted annotation {annotation_id} from {image_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting annotation: {e}")
            return False
    
    def save_annotations_to_file(self, format_type: str = 'json') -> str:
        """Save all annotations to file (JSON or CSV)"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format_type.lower() == 'json':
                filename = f"annotations_{timestamp}.json"
                file_path = Path(self.app.config['ANNOTATIONS_FOLDER']) / filename
                
                # Prepare data for JSON export
                export_data = {
                    'session_id': self.current_session,
                    'created_at': datetime.now().isoformat(),
                    'total_images': len(self.annotations),
                    'total_annotations': sum(len(anns) for anns in self.annotations.values()),
                    'annotations': self.annotations
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                    
            elif format_type.lower() == 'csv':
                filename = f"annotations_{timestamp}.csv"
                file_path = Path(self.app.config['ANNOTATIONS_FOLDER']) / filename
                
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow([
                        'image_filename', 'annotation_id', 'type', 'label', 
                        'coordinates', 'color', 'opacity', 'created_at', 'session_id'
                    ])
                    
                    # Write data
                    for image_filename, annotations in self.annotations.items():
                        for ann in annotations:
                            writer.writerow([
                                image_filename,
                                ann['id'],
                                ann['type'],
                                ann['label'],
                                json.dumps(ann['coordinates']),  # Serialize coordinates as JSON
                                ann['color'],
                                ann['opacity'],
                                ann['created_at'],
                                ann['session_id']
                            ])
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            logger.info(f"Saved annotations to {filename}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving annotations to file: {e}")
            return ""
    
    def load_annotations_from_file(self, file_path: str) -> bool:
        """Load annotations from a previously saved file"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"Annotation file does not exist: {file_path}")
                return False
            
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.annotations = data.get('annotations', {})
                    
            elif file_path.suffix.lower() == '.csv':
                self.annotations = {}
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        image_filename = row['image_filename']
                        if image_filename not in self.annotations:
                            self.annotations[image_filename] = []
                        
                        annotation = {
                            'id': row['annotation_id'],
                            'type': row['type'],
                            'label': row['label'],
                            'coordinates': json.loads(row['coordinates']),
                            'color': row['color'],
                            'opacity': int(row['opacity']),
                            'created_at': row['created_at'],
                            'session_id': row['session_id']
                        }
                        self.annotations[image_filename].append(annotation)
            
            logger.info(f"Loaded annotations from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading annotations from file: {e}")
            return False

    def auto_annotate_image(self, image_filename: str) -> bool:
        """Automatically annotate an image using computer vision"""
        try:
            image_path = Path(self.app.config['AERIAL_IMAGES_FOLDER']) / image_filename
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                return False
            
            # Detect features
            annotations = self.auto_annotator.detect_features(str(image_path))
            
            # Save annotations
            if image_filename not in self.annotations:
                self.annotations[image_filename] = []
            
            # Add metadata to each annotation
            for ann in annotations:
                ann['id'] = str(uuid.uuid4())
                ann['created_at'] = datetime.now().isoformat()
                ann['session_id'] = self.current_session
                ann['auto_generated'] = True
                
                self.annotations[image_filename].append(ann)
            
            logger.info(f"Auto-annotated {image_filename} with {len(annotations)} features")
            return True
            
        except Exception as e:
            logger.error(f"Error auto-annotating {image_filename}: {e}")
            return False

    def auto_annotate_all(self) -> Dict[str, int]:
        """Automatically annotate all images in the aerial images folder"""
        results = {'success': 0, 'failed': 0}
        
        aerial_images = self.get_supported_images(self.app.config['AERIAL_IMAGES_FOLDER'])
        
        for img_filename in aerial_images:
            success = self.auto_annotate_image(img_filename)
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
        
        logger.info(f"Auto-annotation completed: {results['success']} succeeded, {results['failed']} failed")
        return results


# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# Configuration
app.config.update(
    UPLOAD_FOLDER='static/uploads',
    AERIAL_IMAGES_FOLDER='static/aerial_images',
    ANNOTATIONS_FOLDER='annotations',
    THUMBNAILS_FOLDER='static/thumbnails',
    MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50MB max file size
    ALLOWED_EXTENSIONS={'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'}
)

# Initialize annotation app
annotation_app = ImageAnnotationApp(app)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Main page - display annotation interface"""
    # Get list of available aerial images
    aerial_images = annotation_app.get_supported_images(app.config['AERIAL_IMAGES_FOLDER'])
    
    # Get image information for each aerial image
    images_info = []
    for img_filename in aerial_images:
        info = annotation_app.get_image_info(img_filename)
        if info:
            images_info.append(info)
    
    return render_template('index.html', 
                         aerial_images=images_info,
                         total_images=len(images_info))

@app.route('/api/images')
def get_images():
    """API endpoint to get list of available images"""
    try:
        aerial_images = annotation_app.get_supported_images(app.config['AERIAL_IMAGES_FOLDER'])
        images_info = []
        
        for img_filename in aerial_images:
            info = annotation_app.get_image_info(img_filename)
            if info:
                images_info.append(info)
        
        return jsonify({
            'success': True,
            'images': images_info,
            'total': len(images_info)
        })
        
    except Exception as e:
        logger.error(f"Error getting images: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/image/<filename>')
def get_image(filename):
    """Serve an aerial image"""
    try:
        secure_name = secure_filename(filename)
        image_path = Path(app.config['AERIAL_IMAGES_FOLDER']) / secure_name
        
        if not image_path.exists():
            return jsonify({'error': 'Image not found'}), 404
        
        return send_file(image_path)
        
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        return jsonify({'error': 'Error serving image'}), 500

@app.route('/api/thumbnail/<filename>')
def get_thumbnail(filename):
    """Serve a thumbnail image"""
    try:
        thumbnail_path = Path(app.config['THUMBNAILS_FOLDER']) / filename
        
        if not thumbnail_path.exists():
            return jsonify({'error': 'Thumbnail not found'}), 404
        
        return send_file(thumbnail_path)
        
    except Exception as e:
        logger.error(f"Error serving thumbnail {filename}: {e}")
        return jsonify({'error': 'Error serving thumbnail'}), 500

@app.route('/api/annotations/<image_filename>')
def get_annotations(image_filename):
    """Get annotations for a specific image"""
    try:
        secure_name = secure_filename(image_filename)
        annotations = annotation_app.annotations.get(secure_name, [])
        
        return jsonify({
            'success': True,
            'image_filename': secure_name,
            'annotations': annotations,
            'count': len(annotations)
        })
        
    except Exception as e:
        logger.error(f"Error getting annotations for {image_filename}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/annotations/<image_filename>', methods=['POST'])
def save_annotation(image_filename):
    """Save annotation for a specific image"""
    try:
        secure_name = secure_filename(image_filename)
        annotation_data = request.get_json()
        
        if not annotation_data:
            return jsonify({'success': False, 'error': 'No annotation data provided'}), 400
        
        success = annotation_app.save_annotation(secure_name, annotation_data)
        
        if success:
            return jsonify({
                'success': True, 
                'message': 'Annotation saved successfully',
                'total_annotations': len(annotation_app.annotations.get(secure_name, []))
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to save annotation'}), 500
            
    except Exception as e:
        logger.error(f"Error saving annotation for {image_filename}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/annotations/<image_filename>/<annotation_id>', methods=['DELETE'])
def delete_annotation(image_filename, annotation_id):
    """Delete a specific annotation"""
    try:
        secure_name = secure_filename(image_filename)
        success = annotation_app.delete_annotation(secure_name, annotation_id)
        
        if success:
            return jsonify({'success': True, 'message': 'Annotation deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Annotation not found'}), 404
            
    except Exception as e:
        logger.error(f"Error deleting annotation: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_files():
    """Upload new aerial images"""
    try:
        if 'files' not in request.files:
            flash('No files selected')
            return redirect(url_for('index'))
        
        files = request.files.getlist('files')
        uploaded_count = 0
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Avoid filename conflicts
                counter = 1
                original_name = filename
                while Path(app.config['AERIAL_IMAGES_FOLDER'], filename).exists():
                    name, ext = original_name.rsplit('.', 1)
                    filename = f"{name}_{counter}.{ext}"
                    counter += 1
                
                file_path = Path(app.config['AERIAL_IMAGES_FOLDER']) / filename
                file.save(file_path)
                uploaded_count += 1
                logger.info(f"Uploaded aerial image: {filename}")
        
        if uploaded_count > 0:
            flash(f'Successfully uploaded {uploaded_count} image(s)')
        else:
            flash('No valid images were uploaded')
            
        return redirect(url_for('index'))
        
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        flash('Error uploading files')
        return redirect(url_for('index'))

@app.route('/api/export/<format_type>')
def export_annotations(format_type):
    """Export all annotations to JSON or CSV"""
    try:
        if format_type.lower() not in ['json', 'csv']:
            return jsonify({'error': 'Invalid format. Use json or csv'}), 400
        
        file_path = annotation_app.save_annotations_to_file(format_type)
        
        if file_path:
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'Failed to export annotations'}), 500
            
    except Exception as e:
        logger.error(f"Error exporting annotations: {e}")
        return jsonify({'error': 'Error exporting annotations'}), 500

@app.route('/api/import', methods=['POST'])
def import_annotations():
    """Import annotations from uploaded file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.lower().endswith(('.json', '.csv')):
            filename = secure_filename(file.filename)
            temp_path = Path(app.config['ANNOTATIONS_FOLDER']) / f"temp_{filename}"
            file.save(temp_path)
            
            success = annotation_app.load_annotations_from_file(temp_path)
            
            # Clean up temp file
            temp_path.unlink(missing_ok=True)
            
            if success:
                total_annotations = sum(len(anns) for anns in annotation_app.annotations.values())
                return jsonify({
                    'success': True,
                    'message': f'Successfully imported annotations for {len(annotation_app.annotations)} images',
                    'total_annotations': total_annotations
                })
            else:
                return jsonify({'error': 'Failed to import annotations'}), 500
        else:
            return jsonify({'error': 'Invalid file format. Use .json or .csv'}), 400
            
    except Exception as e:
        logger.error(f"Error importing annotations: {e}")
        return jsonify({'error': 'Error importing annotations'}), 500

@app.route('/api/statistics')
def get_statistics():
    """Get annotation statistics"""
    try:
        total_images = len(annotation_app.annotations)
        total_annotations = sum(len(anns) for anns in annotation_app.annotations.values())
        
        # Count by annotation type
        type_counts = {}
        for annotations in annotation_app.annotations.values():
            for ann in annotations:
                ann_type = ann['type']
                type_counts[ann_type] = type_counts.get(ann_type, 0) + 1
        
        # Get images with most annotations
        image_stats = []
        for img_name, annotations in annotation_app.annotations.items():
            if annotations:  # Only include images with annotations
                image_stats.append({
                    'filename': img_name,
                    'annotation_count': len(annotations),
                    'last_annotated': max(ann['created_at'] for ann in annotations)
                })
        
        image_stats.sort(key=lambda x: x['annotation_count'], reverse=True)
        
        return jsonify({
            'success': True,
            'statistics': {
                'total_images': total_images,
                'total_annotations': total_annotations,
                'annotation_types': type_counts,
                'session_id': annotation_app.current_session,
                'top_annotated_images': image_stats[:10]  # Top 10
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auto-annotate/<image_filename>', methods=['POST'])
def auto_annotate_image(image_filename):
    """API endpoint to automatically annotate a specific image"""
    try:
        secure_name = secure_filename(image_filename)
        success = annotation_app.auto_annotate_image(secure_name)
        
        if success:
            annotations = annotation_app.annotations.get(secure_name, [])
            return jsonify({
                'success': True, 
                'message': f'Auto-annotated {secure_name} with {len(annotations)} features',
                'annotation_count': len(annotations)
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to auto-annotate image'}), 500
            
    except Exception as e:
        logger.error(f"Error auto-annotating {image_filename}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auto-annotate-all', methods=['POST'])
def auto_annotate_all():
    """API endpoint to automatically annotate all images"""
    try:
        results = annotation_app.auto_annotate_all()
        
        total_annotations = sum(
            len(anns) for anns in annotation_app.annotations.values()
        )
        
        return jsonify({
            'success': True,
            'message': f'Auto-annotated {results["success"]} images, {results["failed"]} failed',
            'results': results,
            'total_annotations': total_annotations
        })
            
    except Exception as e:
        logger.error(f"Error auto-annotating all images: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File is too large. Maximum size is 50MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found errors"""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories on startup
    print("🚀 Starting Image Annotation App with Auto-Annotation...")
    
    # Check if aerial_images directory exists and has images
    aerial_dir = Path(app.config['AERIAL_IMAGES_FOLDER'])
    if not aerial_dir.exists():
        print(f"📁 Creating aerial images directory: {aerial_dir}")
        aerial_dir.mkdir(parents=True, exist_ok=True)
        print("💡 Add your aerial images to the 'static/aerial_images' directory")
    else:
        image_count = len(annotation_app.get_supported_images(str(aerial_dir)))
        print(f"📸 Found {image_count} aerial images in {aerial_dir}")
    
    print("🌐 App will be available at: http://localhost:5000")
    print("🎨 Ready for manual and automatic image annotation!")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)