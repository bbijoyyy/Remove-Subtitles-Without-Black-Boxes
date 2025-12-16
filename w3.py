import cv2

def simple_video_copy(input_file, output_file):
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
        
        # Display frame
        cv2.imshow('Video', frame)
        
        # Write frame to output
        out.write(frame)
        
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
input_video = "test vid.mp4"
output_video = "output_video.mp4"

simple_video_copy(input_video, output_video)