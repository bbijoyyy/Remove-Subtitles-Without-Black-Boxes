import cv2
import numpy as np

def find_subtitle_box(image_path):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    # Get dimensions to define the "bottom" of the screen later
    height, width, _ = img.shape
    
    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Apply Thresholding
    
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)

    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 45)) 
    
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # 4. Find Contours
    # RETR_EXTERNAL ensures we only get the outer outline, not the holes inside letters like 'O' or 'A'.
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    max_area = 0

    # Define the "Bottom ROI" (Region of Interest)
    # We only care about contours that start in the bottom 30% of the screen
    roi_y_threshold = int(height * 0.7) 

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter 1: Must be in the bottom portion of the screen
        if y > roi_y_threshold:
            
            # Filter 2: Find the largest contour by area
            area = w * h
            if area > max_area:
                max_area = area
                best_contour = cnt

    # 5. Draw the result
    if best_contour is not None:
        x, y, w, h = cv2.boundingRect(best_contour)
        
        # Draw a Green Rectangle (0, 255, 0) with thickness 2
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add label
        cv2.putText(img, "Subtitle ROI", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        print(f"Subtitle found at: x={x}, y={y}, w={w}, h={h}")
    else:
        print("No suitable subtitle contour found in the bottom region.")

    # Show results
    cv2.imshow("Original with Box", img)
    cv2.imwrite("output1.png",img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


find_subtitle_box('input_image1.png')