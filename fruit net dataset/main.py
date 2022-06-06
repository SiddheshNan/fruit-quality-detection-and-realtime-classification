import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Loading..")

import tensorflow.keras
import numpy as np
import cv2
import sys
import tkinter
from tkinter import messagebox
from tkinter import filedialog


main_win = tkinter.Tk()
main_win.geometry("300x100")

main_win.sourceFile = ''


initdir = "/mnt/WD1/python-prj/fruit net dataset/dataset/orange_bad"

def chooseFile():
    main_win.sourceFile = filedialog.askopenfilename(
        parent=main_win, initialdir=initdir, title='Please select a file')
    if main_win.sourceFile:
        main_win.destroy()


b_chooseFile = tkinter.Button(
    main_win, text="Chose File", width=20, height=3, command=chooseFile)
b_chooseFile.place(x=75, y=20)
b_chooseFile.width = 100

main_win.mainloop()


# print(main_win.sourceFolder)

if main_win.sourceFile:
    # print(main_win.sourceFile)
    imgpath = main_win.sourceFile
else:
    print("invalid filepath")
    exit()

# import svm

actions = ['tomato_good',
           'tomato_bad',
           'orange_good',
           'orange_bad'
           ]

print("please wait...")
np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('model.h5', compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


def detect(img):
    img = cv2.resize(img, (224, 224))
    image_array = np.asarray(img)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    # print(prediction)
    prediction_new = prediction[0].tolist()
    detected_action = prediction_new.index(max(prediction_new))
    print("Detected: " + actions[detected_action])
    detected_acc = max(prediction_new)
    print("Quality of Fruit: " + str(detected_acc))
    return actions[detected_action], str(round(detected_acc, 2))


def do_start():
    cv2.namedWindow("Image")
    frame = cv2.imread(imgpath)
    frame = cv2.flip(frame, 1, 1)
    frame = cv2.resize(frame, (224, 224))

    frame = cv2.flip(frame, 1, 1)
    # svm_predcition, svm_coordinates = svm.detectImg(frame)

    det_action, det_accu = detect(frame)
    text = det_action + " - " + det_accu
    cv2.putText(frame, text, (2, 30),
              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Image", frame)

    cv2.waitKey(5000)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    do_start()
