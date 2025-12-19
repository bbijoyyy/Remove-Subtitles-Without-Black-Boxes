import cv2
import numpy as np

def find_subtitle_box(image_path):

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    
    height, width, _ = img.shape
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)

    
    kernel = np.ones((3, 30), np.uint8) 
    dilated = cv2.dilate(thresh, kernel, iterations=1) 
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_y_threshold = int(height * 0.7) 


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
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("Original with Box", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


find_subtitle_box('input_image2.png')