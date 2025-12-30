import cv2
import numpy as np
import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama

# ---------------------------
# Torch CPU-only safety (Remove this if you want to use your GPU)
# ---------------------------
_original_load = torch.jit.load

def cpu_only_load(*args, **kwargs):
    kwargs["map_location"] = "cpu"
    return _original_load(*args, **kwargs)

torch.jit.load = cpu_only_load
torch.set_grad_enabled(False)

lama = SimpleLama()

previous_mask = None
previous_result = None


def frame_change(img) :
    global previous_mask, previous_result
  
    height, width, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------------------------
    # Hard threshold for anime subtitles
    # ---------------------------
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

    # ---------------------------
    # Process only bottom region (but don't mask it blindly)
    # ---------------------------
    roi_y_threshold = int(height * 0.7)
    thresh[:roi_y_threshold, :] = 0

    # ---------------------------
    # Horizontal dilation to connect subtitle letters
    # ---------------------------
    kernel = np.ones((3, 60), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # ---------------------------
    # Find contours
    # ---------------------------
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros((height, width), dtype=np.uint8)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        area = w * h
        aspect_ratio = w / float(h + 1)

        # Strong subtitle filters
        if area > 4000 and aspect_ratio > 5 and y > roi_y_threshold:
            pad_w = 15
            pad_h = 5

            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(width, x + w + pad_w)
            y2 = min(height, y + h + pad_h)

            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # ---------------------------
    # Merge subtitle rectangles vertically (not entire bottom!)
    # ---------------------------
    merge_kernel = np.ones((13, 25), np.uint8)
    mask = cv2.dilate(mask, merge_kernel, iterations=1)

    mask = (mask > 0).astype(np.uint8) * 255

    # cv2.imshow("Mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # ---------------------------
    # Skip LaMa if mask is empty (no subtitles)
    # ---------------------------
    if np.count_nonzero(mask) < 1000:

        return img

    # ---------------------------
    # Resize before LaMa (speed-up)
    # ---------------------------
    MAX_WIDTH = 960

    if width > MAX_WIDTH:
        scale = MAX_WIDTH / width
        img_small = cv2.resize(img, None, fx=scale, fy=scale)
        mask_small = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    else:
        img_small = img
        mask_small = mask



    # ---------------------------
    # Inpainting
    # ---------------------------
    rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb)
    mask_pil = Image.fromarray(mask_small).convert("L")

    
    result_pil = lama(img_pil, mask_pil)
    result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

    if width > MAX_WIDTH:
        result = cv2.resize(result, (width, height))

    # ---------------------------
    # Output
    # ---------------------------
    # cv2.imwrite("sub_removed.png", result)
    # cv2.imshow("Result", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return result

def remove_subtitle(input_file, output_file):
    # Open input video
    cap = cv2.VideoCapture(input_file)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        clean_img=frame_change(frame)

        # Display frame
        cv2.imshow('Video', clean_img)
        
        # Write frame to output
        out.write(clean_img)
        
        frame_count += 1
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Processed {frame_count} frames")
    print(f"Output saved to {output_file}")

# Usage
input_video = "Input_video.mp4"
output_video = "Output_video_DeepLearning.mp4"

remove_subtitle(input_video, output_video)


