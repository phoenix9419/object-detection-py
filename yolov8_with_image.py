import random
import cv2
import numpy as np
from ultralytics import YOLO

# Opening the file in read mode
my_file = open("utils/coco.txt", "r")
# Reading the file
data = my_file.read()
# Replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

# Path to the image
image_path = r'C:\Users\Evenmore\Downloads\learning\object_detection\inference\images\cat.jpeg'

# Read the image
image = cv2.imread(image_path)

# Check if the image was successfully opened
if image is None:
    print("Could not open or find the image.")
    exit()

# Predict on image
detect_params = model.predict(source=[image], conf=0.45, save=False)

# Convert tensor array to numpy
DP = detect_params[0].numpy()
print(DP)

if len(DP) != 0:
    for i in range(len(detect_params[0])):
        boxes = detect_params[0].boxes
        box = boxes[i]  # Returns one box
        clsID = box.cls.numpy()[0]
        conf = box.conf.numpy()[0]
        bb = box.xyxy.numpy()[0]

        cv2.rectangle(
            image,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            detection_colors[int(clsID)],
            3,
        )

        # Display class name and confidence
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(
            image,
            class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
            (int(bb[0]), int(bb[1]) - 10),
            font,
            1,
            (255, 255, 255),
            2,
        )

# Display the resulting image
cv2.imshow("ObjectDetection", image)

# Wait for a key press and close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
