import cv2
import time
import sys
import os
from pathlib import Path

# Add root directory to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.hands import HandPipeline

def draw_landmarks(image, landmarks, color=(0, 255, 0)):
    if landmarks is None:
        return
        
    # MediaPipe Hand Connections
    connections = [
        (0,1), (1,2), (2,3), (3,4), # Thumb
        (0,5), (5,6), (6,7), (7,8), # Index
        (5,9), (9,10), (10,11), (11,12), # Middle
        (9,13), (13,14), (14,15), (15,16), # Ring
        (13,17), (17,18), (18,19), (19,20), # Pinky
        (0,17) # Palm base
    ]
    
    for connection in connections:
        p1 = (int(landmarks[connection[0]][0]), int(landmarks[connection[0]][1]))
        p2 = (int(landmarks[connection[1]][0]), int(landmarks[connection[1]][1]))
        cv2.line(image, p1, p2, color, 2)
        
    for i, lm in enumerate(landmarks):
        px, py = int(lm[0]), int(lm[1])
        cv2.circle(image, (px, py), 4, (255, 0, 0), -1)

def main():
    print("Loading Hand Pipeline (CPU by default since models might be missing)...")
    pipeline = HandPipeline(device="cpu")
    pipeline.load()
    
    # If the models are missing, cv2 video capture will still open but won't draw hands.
    # It will print the error from Hands pipeline and just show the video feed.
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    print("Press 'q' to quit.")
    
    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip horizontally for selfie view
        frame = cv2.flip(frame, 1)
        
        # Run inference
        result = pipeline.infer({"image": frame})
        
        # Draw landmarks
        if result["left"] is not None:
            draw_landmarks(frame, result["left"], (0, 255, 0)) # Green for Left
            cv2.putText(frame, "Left Hand", (int(result["left"][0][0]), int(result["left"][0][1])-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        if result["right"] is not None:
            draw_landmarks(frame, result["right"], (0, 0, 255)) # Red for Right
            cv2.putText(frame, "Right Hand", (int(result["right"][0][0]), int(result["right"][0][1])-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        cv2.imshow("Samvaad Hand Pipeline Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
