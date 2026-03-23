import tkinter as tk
from ultralytics import YOLO
import cv2

def start_detection():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame = results[0].plot()

        cv2.imshow("AI Object Detection", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

app = tk.Tk()
app.title("AI Object Detection App")
app.geometry("300x200")

label = tk.Label(app, text="AI Object Detection", font=("Arial",16))
label.pack(pady=20)

btn = tk.Button(app, text="Start Camera", command=start_detection)
btn.pack(pady=20)

app.mainloop()