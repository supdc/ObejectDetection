import cv2
import datetime

count = 0
prev_object_count = 0

classNames = []
classFile = "D:/embedded pyhon/opencv/project/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "D:/embedded pyhon/opencv/project/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "D:/embedded pyhon/opencv/project/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=[]):
    global count, prev_object_count
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)

    if len(objects) == 0:
        objects = classNames
    objectInfo = []

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    # Draw free space rectangle with black background, adding more space
                    (text_width, text_height), baseline = cv2.getTextSize(classNames[classId - 1].upper(), cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
                    space_height = 10  # Increase this value to add more space
                    cv2.rectangle(img, (box[0] + 10, box[1] + 30 - text_height - space_height), (box[0] + 10 + text_width, box[1] + 30), (0, 0, 0), cv2.FILLED)

                    # Draw text in white with more space
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30 - space_height),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)  # White text
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30 - space_height),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)  # White text

                    # Draw rectangle around the object with green color
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

        # Add timestamp to the image
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, timestamp, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

        if len(classIds) != prev_object_count:
            count += 1  # Increment sample face image
            filename = f"images/E{count}.jpg"
            cv2.imwrite(filename, img)
            prev_object_count = len(classIds)

    return img, objectInfo

if __name__ == "__main__":
    cap = cv2.VideoCapture("http://192.168.208.181:4747/video")
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img, 0.45, 0.2)
        cv2.imshow("Output", img)

        if cv2.waitKey(1) == 32:  # spacebar key is pressed, break from the loop
            break
        elif count >= 100:  # If image taken reaches 100, stop taking video
            break

    cap.release()  # cleanup the camera
    cv2.destroyAllWindows()  # close any open windows
