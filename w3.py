import cv2
import numpy as np

## hello bijoy

def frame_change(img) :

    height, width, _ = img.shape
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

   
    
    kernel = np.ones((3, 60), np.uint8) 
    dilated = cv2.dilate(thresh, kernel, iterations=1) 

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_y_threshold = int(height * 0.7) 

    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
    
        # Filter 1: Must be in the bottom portion
        if y > roi_y_threshold:
        
            # Filter 2: Area Threshold (Instead of "Max Area")
            # We check if the box is big enough to be a subtitle (e.g. > 400 pixels)
            area = w * h
            if area > 4000: 
            
                # Filter 3: Aspect Ratio (Optional but recommended)
                # Subtitles are wide. If w > h, it's likely text.
                if w > h:
                    # Draw the box immediately for THIS contour

                    pad_w=15
                    pad_h=5
                    cv2.rectangle(mask, (x - pad_w, y - pad_h), (x + w + pad_w, y + h + pad_h), 255, -1)

    merge_kernel = np.ones((5, 20), np.uint8) 
    mask = cv2.dilate(mask, merge_kernel, iterations=2)
    clean_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
    
    # cv2.imshow("Original with Box", clean_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return clean_img

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
output_video = "Output_video.mp4"

remove_subtitle(input_video, output_video)

