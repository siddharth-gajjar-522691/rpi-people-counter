import cv2
import numpy as np

# --- CONFIGURATION ---
LINE_COORD = 240  # Y-coordinate of the horizontal line (middle of 480px)
OFFSET = 10       # Pixel buffer to count a "crossing"
CONFIDENCE_THRESHOLD = 0.4 # Slightly lower to catch partial heads

class PersonTracker:
    def __init__(self):
        self.center_points = {} # ID: (x, y)
        self.id_count = 0
        self.total_count = 0
        self.crossed_ids = set()

    def update(self, rects):
        objects_rects = []
        for rect in rects:
            (x, y, w, h) = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            
            # Check if this centroid crosses the line
            # If moving from top (y < 240) to bottom (y > 240)
            if cy > LINE_COORD and cy < (LINE_COORD + OFFSET):
                # We need a way to ensure we only count this ID once
                # Simplified for this example:
                pass 
        
        return objects_rects

# Load model as before...
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
cap = cv2.VideoCapture(0)

# Tracking variables
tracker = {} # stores last known Y position of an ID
total_people = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    (h, w) = frame.shape[:2]
    # Draw the "Invisible" Line (visible for debugging)
    cv2.line(frame, (0, LINE_COORD), (w, LINE_COORD), (0, 255, 255), 2)

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            if idx == 15: # Person
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Calculate Centroid (Middle of the head/body)
                cy = int((startY + endY) / 2)
                cx = int((startX + endX) / 2)

                # TRIGGER LOGIC: If center point crosses the line
                if (LINE_COORD - OFFSET) < cy < (LINE_COORD + OFFSET):
                    # In a real app, you'd use a Centroid Tracker to give IDs.
                    # For a simple solution: we count if it's in this 'hit zone'
                    # Note: This simple version might multi-count without a proper Tracker ID
                    total_people += 1 
                    cv2.line(frame, (0, LINE_COORD), (w, LINE_COORD), (0, 0, 255), 5) # Flash Red

                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.putText(frame, f"Total Count: {total_people}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Overhead Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()