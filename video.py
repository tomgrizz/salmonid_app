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



# Set up logging
logging.basicConfig(filename='video_debug.log', level=logging.DEBUG, 
                    format='%(asctime)s - FRAME %(message)s', filemode='w')


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
        List of dicts with track info for each detected object.
    """

    id2label = {0: 'Chinook', 1: 'Coho', 2: 'Atlantic', 3: 'Rainbow Trout', 4: 'Brown Trout'}
    label2id = {'Chinook': 0, 'Coho': 1, 'Atlantic': 2, 'Rainbow Trout': 3, 'Brown Trout': 4}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_size = (width, height)

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

    for frame_idx in tqdm(range(total_frames), desc=f"Processing {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break

        objects_detected['final_frame_for_video'] = frame_idx

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        if save_video:
            draw = ImageDraw.Draw(pil_image)

        processed = image_processor(pil_image, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, return_dict=True)

        detections = image_processor.post_process_object_detection(
                outputs, target_sizes=torch.tensor([[max(pil_image.size), max(pil_image.size)]]), threshold=box_score_thresh
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
    


    counts = {}
    for id in objects_detected.keys():
        if id == 'final_frame_for_video':
            continue

        counts[id] = {}
        first_x1, first_y1, first_x2, first_y2 = [int(x) for x in objects_detected[id]['boxes'][0]]
        last_x1, last_y1, last_x2, last_y2 = [int(x) for x in objects_detected[id]['boxes'][-1]]
        
        # Note entrance sice (if any)
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


    
    return counts
        