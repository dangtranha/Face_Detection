import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
import os
import time
import sqlite3
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier('haar_cascade_files/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

window = tk.Tk()
window.geometry('300x100')
window.title("Face Recognition App")

def create_data():
    cap = cv2.VideoCapture(0)
    img_counter = 0
    name = simpledialog.askstring("Input", "Enter the name of the person:")
    if not name:
        return

    connection = sqlite3.connect('Person.db')
    cursor = connection.cursor()
    cursor.execute("PRAGMA table_info('Persons')")
    table_exists = cursor.fetchall()
    if not table_exists:
        cursor.execute('''CREATE TABLE Persons(
            Id INTEGER,
            Name TEXT
        )''')
        connection.commit()
        print("'Persons' created.")
    else:
        print("'Persons' already exists.")
    cursor.execute("SELECT MAX(Id) FROM Persons")
    max_id = cursor.fetchone()[0]
    if max_id is None:
        max_id = 0
    max_id += 1

    dir_name = "dataset/"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    while img_counter < 30:
        ret, img = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            img_name = dir_name + "/{0}_{1}.png".format(max_id, time.time())
            cv2.imwrite(img_name, roi_gray)
            img_counter += 1
        cv2.imshow('Creating Data', img)
        if cv2.waitKey(30) & 0xFF == 27:
            break
        time.sleep(0.1)

    if img_counter > 0:
        sql_str = "INSERT INTO Persons (Id, Name) VALUES (?, ?)"
        cursor.execute(sql_str, (max_id, name))
        connection.commit()

    connection.close()
    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", "Data creation completed successfully!")
    train_recognizer()

def train_recognizer():
    path = 'dataset'

    def get_images_and_labels(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]
        face_samples = []
        ids = []

        for image_path in image_paths:
            pil_image = Image.open(image_path).convert('L')  # convert it to grayscale
            img_numpy = np.array(pil_image, 'uint8')
            id = int(os.path.split(image_path)[-1].split('_')[0])
            faces = face_cascade.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)
        
        return face_samples, ids

    print("\n[INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = get_images_and_labels(path)
    recognizer.train(faces, np.array(ids))

    if not os.path.exists('model'):
        os.makedirs('model')
    recognizer.save('model/trainer.yml')
    print(f"\n[INFO] {len(np.unique(ids))} faces trained. Exiting Program.")

def face_recognition():
    recognizer.read('model/trainer.yml')

    connection = sqlite3.connect('Person.db')
    cursor = connection.cursor()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 100:
                cursor.execute("SELECT Name FROM Persons WHERE Id = ?", (id,))
                result = cursor.fetchone()
                name = result[0] if result else "Unknown"
                confidence_text = f"  {round(100 - confidence)}%"
            else:
                name = "Unknown"
                confidence_text = f"  {round(100 - confidence)}%"
            cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, confidence_text, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    connection.close()
    cv2.destroyAllWindows()

btn_create_data = tk.Button(window, text="Add Data", command=create_data)
btn_create_data.pack(expand=True, side="left")
btn_face_recognition = tk.Button(window, text="Face Recognition", command=face_recognition)
btn_face_recognition.pack(expand=True, side= "left")

window.mainloop()
