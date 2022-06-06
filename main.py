import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import svm
import sys
import cv2
import numpy as np
import tensorflow.keras
from imutils.video import VideoStream
import imutils
import time
import serial

port = "/tty/USB0"

check = 30
motor_delay = 15

CAMERA_INDEX = 0

lables = open("lables.txt", "r").read()
actions = lables.split("\n")

print("[INFO] loading...")
np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('fruit.model', compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

print("[INFO] starting video stream...")
vs = VideoStream(src=CAMERA_INDEX).start()
arduino = serial.Serial(port, 9600)
time.sleep(2.0)



def detect(img):
    # img = cv2.resize(img, (224, 224))
    image_array = np.asarray(img)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    # print(prediction)
    prediction_new = prediction[0].tolist()
    detected_action = prediction_new.index(max(prediction_new))
    print("Detected: " + actions[detected_action])
    detected_acc = max(prediction_new)
    print("Quality of fruit: " + str(detected_acc))
    return actions[detected_action], str(round(detected_acc, 2))


def do_start():
    cv2.namedWindow("Image")
    count = 0
    last = ""
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        frame = cv2.flip(frame, 1, 1)
        frame = cv2.resize(frame, (224, 224))

        # frame = cv2.flip(frame, 1, 1)
        # svm_predcition, svm_coordinates = svm.detectImg(frame)
        #
        # det_action, det_accu = detect(frame, {'pred': svm_predcition, 'coord': svm_coordinates})

        det_action, det_accu = detect(frame)
        text = det_action + " - " + det_accu
        cv2.putText(frame, text, (2, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if det_action != "empty":

            if last == "" or last == det_action:
                last = det_action
                count += 1
            elif last != det_action:
                last = ""
                count = 0

            if count == check:
                print("Confirmed:", det_action)
                if "good" in det_action:
                    arduino.write('A'.encode())
                if "bad" in det_action:
                    arduino.write('B'.encode())
                time.sleep(motor_delay)
                arduino.write('S'.encode())

        cv2.imshow("Image", frame)

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    do_start()
