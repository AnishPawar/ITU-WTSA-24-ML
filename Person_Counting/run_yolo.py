
import cv2
from ultralytics import YOLO





import argparse

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process some integers and strings.")

    # Add an integer argument
    parser.add_argument('-n', '--capacity', type=int, default=50)

    # Add a string argument
    parser.add_argument('-s', '--model', type=str,default='s')

    parser.add_argument('-f', '--conf', type=float,default=0.3)

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    print(f'Capacity: {args.capacity}')
    print(f'Model: "{args.model}"')
    print(f'Conf: "{args.conf}"')

    # if model == "m":


    model = YOLO(f'yolov8{args.model}.pt')

    video_path = 'bus_video.mp4'
    cap = cv2.VideoCapture(video_path)


    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        results = model(frame,verbose=False)

        person_detections = []
        for result in results[0].boxes:
            confidence = result.conf[0]
            if result.cls[0] == 0 and confidence >= args.conf:
                person_detections.append(result)

        print(f"I've detected: {len(person_detections)} persons | Empty Seats are: {args.capacity - len(person_detections)}")

        annotated_frame = frame.copy()
        for detection in person_detections:
        
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box
            cv2.putText(annotated_frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('YOLOv8 Person Detection', annotated_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()