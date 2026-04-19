import cv2 
import numpy as np
import os 

PROTOTEXT = 'MobileNetSSD_deploy.prototxt'
MODEL = 'MobileNetSSD_deploy.caffemodel'
CONFIDENCE_THRESHOLD = 0.5

# Categories MobileNet can detect (Person is index 15)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

 
def main():
    # Load the serialized model from disk
    print("[INFO] Loading Model")
    net = cv2.dnn.readNetFromCaffe(PROTOTEXT, MODEL)

    # Initialize video stream (o is typically the RPi Cam)
    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(0)
    
    # set lower resolution for RPi Zero W 2 performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        (h, w) = frame.shape[:2]
        
        # Resize for the model (300x300 is standard for MobileNet SSD)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        
        person_count = 0
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > CONFIDENCE_THRESHOLD: # 50% threshold
                idx = int(detections[0, 0, i, 1])
                
                if CLASSES[idx] == "person":
                    person_count += 1
                    
                    # Draw bounding bix logic 
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Draw the prediction on the frame
                    label = f"person :  {confidence * 100:.2f}%"
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255,0),2)
                    cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0,255,0), 2)
                            
        cv2.putText(frame, f"Current Count : {person_count}", (10,30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Real-Time Monitor", frame)
        
        key = cv2.waitKey(1) & 0xFF 
        if key == ord('q'):
            break;

    cap.release()
    cap.destroyAllWindows()


if __name__ == "__main__":
    main()