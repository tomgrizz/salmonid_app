import os
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
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
    track_id_list = []

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

        # DEBUG PRINT - IGNORE
        # if len(res) == 0:
        #     print(f"No tracks created at frame {frame_idx} with scores: {dets[:, 4]}")
        # else:
        #     print(f"Tracks created at frame {frame_idx} with scores: {dets[:, 4]}")

        # Update and draw tracked objects
        for track in res:
            # Unpack track data, keeping confidence as a float
            x1, y1, x2, y2, track_id, conf, cls, _ = track
            x1, y1, x2, y2, track_id, cls = int(x1), int(y1), int(x2), int(y2), int(track_id), int(cls)
            
            track_id_list.append(track_id)

            if track_id not in objects_detected.keys():
                track_dict = {}
                track_dict['first_box'] = [x1, y1, x2, y2]
                track_dict['cls'] = {cls: conf}
                track_dict['first_frame'] = frame_idx
                objects_detected[track_id] = track_dict
            else:
                objects_detected[track_id]['last_frame'] = frame_idx
                objects_detected[track_id]['last_box'] = [x1, y1, x2, y2]
                if cls in objects_detected[track_id]['cls'].keys():
                    objects_detected[track_id]['cls'][cls] += conf
                else:
                    objects_detected[track_id]['cls'][cls] = conf

            objects_detected['track_ids'] = set(track_id_list)
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

    counts = []
    width = max(pil_image.size)
    for track_id in objects_detected.get('track_ids', []):
        id = {}
        id['id'] = int(track_id)
        cls_dict = objects_detected[track_id]['cls']
        cls_idx = int(max(cls_dict, key=lambda k: cls_dict[k]))
        if id2label is not None:
            id['cls'] = id2label.get(cls_idx, str(cls_idx))
        else:
            id['cls'] = cls_idx
        first_x1, first_y1, first_x2, first_y2 = [int(x) for x in objects_detected[track_id]['first_box']]
        last_x1, last_y1, last_x2, last_y2 = [int(x) for x in objects_detected[track_id].get('last_box', objects_detected[track_id]['first_box'])]
        
        # If the first frame is less than 5, we don't know what side the fish entered.
        if objects_detected[track_id]['first_frame'] < 5:
            id['entrance'] = 'None'
        else:
            if (first_x1 + first_x2) / 2 < width * 0.5:
                id['entrance'] = 'left'
            else:
                id['entrance'] = 'right'
        
        # If the last frame is the final frame, we don't know what side the fish exited.
        if objects_detected[track_id].get('last_frame', 0) == objects_detected['final_frame_for_video']:
            id['exit'] = 'None'
        else:
            if (last_x1 + last_x2) / 2 < width * 0.5:
                id['exit'] = 'left'
            else:
                id['exit'] = 'right'
        counts.append(id)
    return counts 