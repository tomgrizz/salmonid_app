# Replace the entire content of video.py with this:

import os
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from collections import Counter
import imageio.v2 as imageio
import logging
import easyocr
import re
from datetime import datetime

# Set up logging
logging.basicConfig(filename='video_debug.log', level=logging.DEBUG, 
                    format='%(asctime)s - FRAME %(message)s', filemode='w')

def extract_timestamp_from_frame(frame, reader):
    """
    Extract timestamp from the top-left corner of a video frame.
    Handles various river names and OCR artifacts.
    """
    try:
        # Crop the top-left region where timestamp is likely to be
        height, width = frame.shape[:2]
        # Try multiple crop regions
        crop_regions = [
            (0, int(height * 0.1), 0, int(width * 0.4)),  # Top 10%, left 40%
            (0, int(height * 0.15), 0, int(width * 0.5)), # Top 15%, left 50%
            (0, int(height * 0.2), 0, int(width * 0.6)),  # Top 20%, left 60%
            (0, int(height * 0.1), 0, int(width * 0.8)),  # Top 10%, left 80%
        ]
        
        for y1, y2, x1, x2 in crop_regions:
            timestamp_region = frame[y1:y2, x1:x2]
            
            # Read text from the region
            results = reader.readtext(timestamp_region)
            
            # Combine all detected text
            full_text = ' '.join([text[1] for text in results])
            
            # Multiple patterns to handle different OCR artifacts and river names
            patterns = [
                # Pattern 1: "OMNRF Ganaraska River 13 24 05.34.21PM May" (exact format we saw)
                r'OMNRF?\s+.*?(\d{1,2})\s+(\d{2})\s+(\d{1,2})[.:](\d{2})[.:](\d{2})(AM|PM)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
                
                # Pattern 2: "OMNR - Credit River 13 May 24 01:16:29PM" (standard format)
                r'OMNR\s*[-]?\s+.*?(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{2})\s+(\d{1,2})[.:](\d{2})[.:](\d{2})(AM|PM)',
                
                # Pattern 3: More flexible pattern for any river name
                r'OMNRF?\s+.*?(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{2})\s+(\d{1,2})[.:](\d{2})[.:](\d{2})(AM|PM)',
                
                # Pattern 4: Handle OCR artifacts like "O" instead of "0"
                r'OMNRF?\s+.*?(\d{1,2})\s+(\d{2})\s+(\d{1,2})[.:](\d{2})[.:](\d{2})(AM|PM)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    try:
                        groups = match.groups()
                        
                        # Handle different pattern formats
                        if len(groups) == 7:
                            if groups[6] in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
                                # Pattern 1 or 4: day, year, hour, minute, second, ampm, month
                                day = int(groups[0])
                                year = int('20' + groups[1])
                                hour = int(groups[2])
                                minute = int(groups[3])
                                second = int(groups[4])
                                ampm = groups[5].upper()
                                month_name = groups[6]
                            else:
                                # Pattern 2 or 3: day, month, year, hour, minute, second, ampm
                                day = int(groups[0])
                                month_name = groups[1]
                                year = int('20' + groups[2])
                                hour = int(groups[3])
                                minute = int(groups[4])
                                second = int(groups[5])
                                ampm = groups[6].upper()
                        
                        # Clean up OCR artifacts (replace 'O' with '0' for numbers)
                        hour = int(str(hour).replace('O', '0'))
                        minute = int(str(minute).replace('O', '0'))
                        second = int(str(second).replace('O', '0'))
                        
                        # Validate ranges
                        if not (1 <= day <= 31 and 1 <= hour <= 12 and 0 <= minute <= 59 and 0 <= second <= 59):
                            continue
                        
                        # Convert to 24-hour format
                        if ampm == 'PM' and hour != 12:
                            hour += 12
                        elif ampm == 'AM' and hour == 12:
                            hour = 0
                        
                        # Create datetime object
                        month_map = {
                            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                        }
                        month = month_map[month_name]
                        
                        # Validate the date
                        dt = datetime(year, month, day, hour, minute, second)
                        print(f"✅ Successfully parsed timestamp: {dt}")
                        return dt
                        
                    except (ValueError, KeyError) as e:
                        print(f"Error parsing timestamp components: {e}")
                        continue
        
        return None
        
    except Exception as e:
        print(f"Error extracting timestamp: {e}")
        return None

def debug_river_detection(frame, reader):
    """
    Debug function to see what river names are being detected.
    """
    try:
        height, width = frame.shape[:2]
        crop_height = int(height * 0.1)
        crop_width = int(width * 0.4)
        
        timestamp_region = frame[0:crop_height, 0:crop_width]
        results = reader.readtext(timestamp_region)
        
        full_text = ' '.join([text[1] for text in results])
        print(f"Full detected text: '{full_text}'")
        
        # Look for river-related keywords
        river_keywords = ['river', 'ganaraska', 'credit', 'omnr', 'omnr']
        found_keywords = [word for word in river_keywords if word.lower() in full_text.lower()]
        
        if found_keywords:
            print(f"Found river keywords: {found_keywords}")
        
        return full_text
        
    except Exception as e:
        print(f"Error in river detection debug: {e}")
        return None

def debug_timestamp_extraction(frame, reader):
    """
    Debug function to see what text is being detected.
    """
    try:
        # Crop the top-left region where timestamp is likely to be
        height, width = frame.shape[:2]
        print(f"Frame dimensions: {width}x{height}")
        
        # Try different crop regions
        crop_regions = [
            (0, int(height * 0.1), 0, int(width * 0.4)),  # Top 10%, left 40%
            (0, int(height * 0.15), 0, int(width * 0.5)), # Top 15%, left 50%
            (0, int(height * 0.2), 0, int(width * 0.6)),  # Top 20%, left 60%
            (0, int(height * 0.1), 0, int(width * 0.8)),  # Top 10%, left 80%
        ]
        
        for i, (y1, y2, x1, x2) in enumerate(crop_regions):
            print(f"\n--- Region {i+1}: ({x1},{y1}) to ({x2},{y2}) ---")
            timestamp_region = frame[y1:y2, x1:x2]
            
            # Read text from the region
            results = reader.readtext(timestamp_region)
            
            if results:
                print(f"Detected text: {results}")
                # Combine all detected text
                full_text = ' '.join([text[1] for text in results])
                print(f"Combined text: '{full_text}'")
                
                # Look for any timestamp-like patterns
                if 'OMNR' in full_text or 'Ganaraska' in full_text or 'River' in full_text:
                    print(f"*** POTENTIAL TIMESTAMP FOUND: {full_text} ***")
                    return full_text
            else:
                print("No text detected in this region")
        
        return None
        
    except Exception as e:
        print(f"Error in debug extraction: {e}")
        return None

def format_timestamp_for_json(dt):
    """
    Format datetime object to the desired JSON format.
    
    Args:
        dt: datetime object
        
    Returns:
        Formatted string like "May 13, 2024 13:16"
    """
    if dt is None:
        return None
    
    return dt.strftime("%B %d, %Y %H:%M")

def calculate_confidence_metrics(objects_detected):
    """
    Calculate confidence metrics for the entire video.
    
    Args:
        objects_detected: Dictionary containing detection data
        
    Returns:
        Dictionary with confidence metrics
    """
    all_confidences = []
    class_confidences = {}
    
    for track_id, track_data in objects_detected.items():
        if track_id == 'final_frame_for_video':
            continue
            
        # Get all confidences for this track
        confidences = track_data['confs']
        classes = track_data['classes']
        
        # Add to overall confidences
        all_confidences.extend(confidences)
        
        # Group by class
        for conf, cls in zip(confidences, classes):
            if cls not in class_confidences:
                class_confidences[cls] = []
            class_confidences[cls].append(conf)
    
    if not all_confidences:
        return {
            'overall_confidence': 0.0,
            'min_confidence': 0.0,
            'max_confidence': 0.0,
            'avg_confidence': 0.0,
            'total_detections': 0,
            'class_confidences': {}
        }
    
    # Calculate overall metrics
    overall_metrics = {
        'overall_confidence': float(np.mean(all_confidences)),
        'min_confidence': float(np.min(all_confidences)),
        'max_confidence': float(np.max(all_confidences)),
        'avg_confidence': float(np.mean(all_confidences)),
        'total_detections': len(all_confidences)
    }
    
    # Calculate per-class metrics
    class_metrics = {}
    for cls, confs in class_confidences.items():
        class_metrics[cls] = {
            'avg_confidence': float(np.mean(confs)),
            'min_confidence': float(np.min(confs)),
            'max_confidence': float(np.max(confs)),
            'detection_count': len(confs)
        }
    
    overall_metrics['class_confidences'] = class_metrics
    
    return overall_metrics

def process_video(
    video_path,
    model,
    image_processor,
    tracker,
    save_video=False,
    output_dir=None,
    device="cuda",
    box_score_thresh=0.1,
    font_path="arial.ttf",
    id2label=None,
):
    """
    Process a video for object detection and tracking.

    Args:
        video_path (str): Path to the video file.
        model: Object detection model.
        image_processor: Image processor for the model.
        tracker: Multi-object tracker instance.
        save_video (bool): Whether to save annotated video output.
        output_dir (str): Directory to save annotated video (if save_video=True).
        device (str): Device to run inference on.
        box_score_thresh (float): Detection score threshold.
        font_path (str): Path to font for annotation.
        id2label (dict): Mapping from class index to class name.

    Returns:
        Dict with counts, timestamp, and confidence info.
    """

    id2label = {0: 'Chinook', 1: 'Coho', 2: 'Atlantic', 3: 'Rainbow Trout', 4: 'Brown Trout'}
    label2id = {'Chinook': 0, 'Coho': 1, 'Atlantic': 2, 'Rainbow Trout': 3, 'Brown Trout': 4}

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_size = (width, height)
    
    # Clear GPU cache at the start of processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Prepare output path for imageio
    if save_video:
        if output_dir is None:
            output_path = os.path.splitext(video_path)[0] + "_annotated.mp4"
        else:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir, os.path.splitext(os.path.basename(video_path))[0] + "_annotated.mp4"
            )
        frames = []

    to_pil = torchvision.transforms.ToPILImage()

    # Load font
    try:
        font = ImageFont.truetype(font_path, 16)
    except IOError:
        font = ImageFont.load_default()

    objects_detected = {}
    timestamp_extracted = None

    for frame_idx in tqdm(range(total_frames), desc=f"Processing {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break

        objects_detected['final_frame_for_video'] = frame_idx

        # Extract timestamp from first few frames (only once)
        if timestamp_extracted is None and frame_idx < 10:
            print(f"\n=== Attempting timestamp extraction on frame {frame_idx} ===")
            debug_text = debug_river_detection(frame, reader)
            if debug_text:
                timestamp_extracted = extract_timestamp_from_frame(frame, reader)
                if timestamp_extracted:
                    print(f"✅ Successfully extracted timestamp: {format_timestamp_for_json(timestamp_extracted)}")
                else:
                    print(f"❌ Text found but couldn't parse timestamp: {debug_text}")
            else:
                print("❌ No text detected in timestamp region")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        if save_video:
            draw = ImageDraw.Draw(pil_image)

        processed = image_processor(pil_image, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device)
        # Move all tensors to the same device as the model
        for key in processed:
            if isinstance(processed[key], torch.Tensor):
                processed[key] = processed[key].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, return_dict=True)

        detections = image_processor.post_process_object_detection(
                outputs, target_sizes=torch.tensor([[max(pil_image.size), max(pil_image.size)]], device=device), threshold=box_score_thresh
            )[0]
        
        if len(detections['boxes']) == 0:
            if save_video:
                frames.append(np.array(pil_image))
            continue
        
        # Only update the tracker if there are detections to avoid a library bug
        boxes = detections['boxes'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        labels = detections['labels'].cpu().numpy()

        dets = np.column_stack((boxes, scores, labels))
        res = tracker.update(dets, frame)

        # Update and draw tracked objects
        for track in res:
            # Unpack track data, keeping confidence as a float
            x1, y1, x2, y2, track_id, conf, cls, _ = track
            x1, y1, x2, y2, track_id, cls = int(x1), int(y1), int(x2), int(y2), int(track_id), int(cls)

            if track_id in objects_detected.keys():
                objects_detected[track_id]['boxes'].append([x1, y1, x2, y2])
                objects_detected[track_id]['classes'].append(cls)
                objects_detected[track_id]['confs'].append(conf)
                objects_detected[track_id]['frames'].append(frame_idx)
            else:
                objects_detected[track_id] = {}
                objects_detected[track_id]['boxes'] = [[x1, y1, x2, y2]]
                objects_detected[track_id]['classes'] = [cls]
                objects_detected[track_id]['confs'] = [conf]
                objects_detected[track_id]['frames'] = [frame_idx]
        
        if save_video:
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
            draw.text((x1, y1 - 20), f"Fish ID: {track_id}", fill="red", font=font)

        # Optional: draw raw boxes (always do this to see what the model detects)
        if save_video:
            # We must use the original detections tensor here, not the numpy array
            for box, score, label in zip(detections['boxes'].cpu(), detections['scores'].cpu(), detections['labels'].cpu()):
                if score > box_score_thresh:
                    x1, y1, x2, y2 = map(int, box)
                    draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)
                    draw.text((x1, y1 - 10), f"Label: {label} ({score:.2f})", fill="green", font=font)

        if save_video:
            annotated_frame = np.array(pil_image)
            frames.append(annotated_frame)

    cap.release()
    if save_video and frames:
        with imageio.get_writer(output_path, fps=fps, codec='libx264') as writer:
            print(f"Saving video to {output_path}")
            for frame in frames:
                writer.append_data(frame)

    # Process tracking results
    counts = {}
    for id in objects_detected.keys():
        if id == 'final_frame_for_video':
            continue

        counts[id] = {}
        first_x1, first_y1, first_x2, first_y2 = [int(x) for x in objects_detected[id]['boxes'][0]]
        last_x1, last_y1, last_x2, last_y2 = [int(x) for x in objects_detected[id]['boxes'][-1]]
        
        # Note entrance side (if any)
        if objects_detected[id]['frames'][0] <= 3:
            counts[id]['entry'] = 'None'
        elif (first_x1 + first_x2) / 2 < width * 0.5:
            counts[id]['entry'] = 'Left'
        elif (first_x1 + first_x2) / 2 > width * 0.5:
            counts[id]['entry'] = 'Right'
        else:
            counts[id]['entry'] = 'None'
        
        # Note exit side (if any)
        if objects_detected[id]['frames'][-1] >= objects_detected['final_frame_for_video'] - 3:
            counts[id]['exit'] = 'None'
        elif (last_x1 + last_x2) / 2 < width * 0.5:
            counts[id]['exit'] = 'Left'
        elif (last_x1 + last_x2) / 2 > width * 0.5:
            counts[id]['exit'] = 'Right'
        else:
            counts[id]['exit'] = 'None'
        
        # Note class of object
        class_list = objects_detected[id]['classes']
        class_counts = Counter(class_list)
        most_common_class = class_counts.most_common(1)[0][0]
        
        counts[id]['Class'] = id2label[most_common_class]
        
        # Add confidence metrics for this track
        confidences = objects_detected[id]['confs']
        counts[id]['avg_confidence'] = float(np.mean(confidences))
        counts[id]['min_confidence'] = float(np.min(confidences))
        counts[id]['max_confidence'] = float(np.max(confidences))

    # Calculate overall confidence metrics
    confidence_metrics = calculate_confidence_metrics(objects_detected)
    
    # Add timestamp to the results
    result = {
        'timestamp': format_timestamp_for_json(timestamp_extracted),
        'confidence_metrics': confidence_metrics,
        'fish_counts': counts
    }
    
    return result