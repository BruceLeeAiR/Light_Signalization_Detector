import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
from ultralytics import YOLO
import random
import os
from PIL import Image, ImageTk
import threading
import subprocess
import platform

random_image_active = False
cap = None
camera_running = False
video_running = False

def show_camera(display_label):
    global cap, camera_running
    camera_running = True
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nie udało się otworzyć kamery!")
        camera_running = False
        return

    model = YOLO("best.pt")

    def camera_loop():
        if not camera_running:
            return  # zatrzymaj pętlę, jeśli GUI zostało zresetowane

        ret, frame = cap.read()
        if not ret or frame is None:
            print("Nie udało się odczytać klatki z kamery!")
            return

        results = model.predict(source=frame, save=False, conf=0.3, verbose=False)

        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            confidence = result.conf[0]
            class_id = int(result.cls[0])
            label = f"{results[0].names[class_id]} {confidence:.2f}"
            draw_box_with_color(frame, label, x1, y1, x2, y2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        display_label.imgtk = imgtk
        display_label.configure(image=imgtk)
        display_label.update_idletasks()
        display_label.after(50, camera_loop)

    camera_loop()

def draw_box_with_color(frame, label, x1, y1, x2, y2):
    color_map = {
        "Backhand": (0, 0, 255),  # Czerwony
        "Forehand": (0, 255, 255),  # Żółty
        "Ready_position": (0, 255, 0),  # Zielony
        "Serve": (255, 255, 0),  # Czarny
        "Tennis_ball": (0, 0, 0),  # Niebieski
    }
    label_name = label.split()[0]  # np. "red_light"

    color = color_map.get(label_name, (255, 255, 255))  # domyślnie biały

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

folder_path = r"C:\Users\jakub\Desktop\Projekt ICR\Baza Zdjecia"

def analyze_random_image(display_label, folder_path):
    global random_image_active
    random_image_active = True

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("Brak obrazów w folderze!")
        return

    random_image_path = os.path.join(folder_path, random.choice(image_files))
    image = cv2.imread(random_image_path)

    model = YOLO("best.pt")
    results = model.predict(source=image, save=False, conf=0.3, verbose=False)

    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        confidence = result.conf[0]
        class_id = int(result.cls[0])
        label = f"{results[0].names[class_id]} {confidence:.2f}"
        draw_box_with_color(image, label, x1, y1, x2, y2)

    # Przekształcenie do RGB (PIL używa RGB, OpenCV BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Przekształcenie na obiekt PIL
    image_pil = Image.fromarray(image_rgb)

    # Pobierz rozmiar display_label (uwzględnij sytuację, gdy jeszcze nie jest wyrenderowany)
    label_width = display_label.winfo_width()
    label_height = display_label.winfo_height()
    if label_width <= 1 or label_height <= 1:
        label_width = 800
        label_height = 600

    # Dopasuj obraz z zachowaniem proporcji (thumbnail robi to automatycznie)
    image_pil.thumbnail((label_width, label_height), Image.Resampling.LANCZOS)

    # Konwertuj do formatu Tkinter
    imgtk = ImageTk.PhotoImage(image=image_pil)

    # Wyświetl w labelu
    display_label.imgtk = imgtk
    display_label.configure(image=imgtk)

def play_video(display_label):
    global cap, stop_flag, video_running
    cap = cv2.VideoCapture(r"C:\Users\jakub\Desktop\Projekt ICR\Analiza.mp4")

    if not cap.isOpened():
        print("Nie można otworzyć pliku wideo!")
        return

    model = YOLO("best.pt")
    video_running = True

    new_width = 640
    new_height = 360
    frame_count = 0

    stop_flag.set(False)

    def process_frame():
        nonlocal frame_count

        if not video_running or stop_flag.get():
            if cap:
                cap.release()
            return

        ret, frame = cap.read()
        if not ret:
            if cap:
                cap.release()
            return

        resized_frame = cv2.resize(frame, (new_width, new_height))

        if frame_count % 5 == 0:
            results = model.predict(source=resized_frame, save=False, conf=0.3, verbose=False)

            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                confidence = result.conf[0]
                class_id = int(result.cls[0])
                label = f"{results[0].names[class_id]} {confidence:.2f}"
                draw_box_with_color(resized_frame, label, x1, y1, x2, y2)

        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        display_label.imgtk = imgtk
        display_label.configure(image=imgtk)
        display_label.update_idletasks()

        frame_count += 1
        display_label.after(500, process_frame)

    process_frame()

def play_analyzed_video(display_label, video_path):
    global cap, video_running
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Nie można otworzyć przeanalizowanego pliku wideo!")
        return

    video_running = True

    # Pobierz FPS z filmu, żeby ustalić realne opóźnienie
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 40  # ms

    # Pobierz rozmiary labela
    label_width = display_label.winfo_width()
    label_height = display_label.winfo_height()
    if label_width <= 1 or label_height <= 1:
        label_width = 800
        label_height = 600

    def update_frame():
        if not video_running:
            if cap:
                cap.release()
            return

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # restart filmu
            ret, frame = cap.read()
            if not ret:
                if cap:
                    cap.release()
                return

        # Dopasuj rozmiar klatki do labela
        frame = cv2.resize(frame, (label_width, label_height))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        display_label.imgtk = imgtk
        display_label.configure(image=imgtk)
        display_label.update_idletasks()

        display_label.after(delay, update_frame)

    update_frame()


def analyze_and_save_video():
    input_path = r"C:\Users\jakub\Desktop\Projekt ICR\Analiza.mp4"
    output_path = os.path.join(os.getcwd(), "output_analyzed_video.mp4")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Nie można otworzyć pliku wideo!")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    model = YOLO("best.pt")
    frame_count = 0
    detection_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_detections = []
        if frame_count % 5 == 0:
            results = model.predict(source=frame, save=False, conf=0.5, verbose=False)
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                confidence = result.conf[0]
                class_id = int(result.cls[0])
                label = f"{results[0].names[class_id]} {confidence:.2f}"
                current_detections.append((x1, y1, x2, y2, label))
                draw_box_with_color(frame, label, x1, y1, x2, y2)
        else:
            if detection_history:
                for detection in detection_history[-1]:
                    x1, y1, x2, y2, label = detection
                    draw_box_with_color(frame, label, x1, y1, x2, y2)

        detection_history.append(current_detections)
        out.write(frame)
        frame_count += 1

        progress = (frame_count / total_frames) * 100
        progress_var.set(progress)
        root.update_idletasks()

    cap.release()
    out.release()
    progress_var.set(100)
    print(f"Zapisano przetworzony film jako: {output_path}")

    # Reset progress bar after short delay
    root.after(2000, lambda: progress_var.set(0))

    # *** Najważniejsza zmiana: WŁĄCZ FILM W LABELU ***
    play_analyzed_video(display_label, output_path)


def reset_gui(display_label):
    global cap, random_image_active, camera_running, video_running
    camera_running = False
    video_running = False
    if cap is not None and cap.isOpened():
        cap.release()
        cap = None
    if random_image_active:
        random_image_active = False
    display_label.configure(image="")
    display_label.imgtk = None

root = tk.Tk()
root.title("TENNIS TECHNIQUE DETECT")
root.geometry("1200x600")
root.configure(bg="#f0f0f5")

stop_flag = tk.BooleanVar(value=False)

title = tk.Label(root, text="TENNIS TECHNIQUE DETECT", font=("Arial", 40, "bold"), bg="#f0f0f5", fg="#333333")
title.pack(pady=20)

button_frame = tk.Frame(root, bg="#f0f0f5")
button_frame.pack(side=tk.LEFT, padx=20, pady=20)

button_style = {
    "bg": "#4a90e2",
    "fg": "white",
    "font": ("Arial", 14, "bold"),
    "width": 20,
    "height": 2,
    "borderwidth": 2
}

btn_reset = tk.Button(button_frame, text="RESET GUI", command=lambda: reset_gui(display_label), **button_style)
btn_reset.pack(pady=10)

btn_video = tk.Button(button_frame, text="PLAY VIDEO", command=lambda: play_video(display_label), **button_style)
btn_video.pack(pady=10)

btn_image = tk.Button(button_frame, text="RANDOM IMAGE", command=lambda: analyze_random_image(display_label, folder_path), **button_style)
btn_image.pack(pady=10)

btn_save_video = tk.Button(button_frame, text="ANALYZE VIDEO", command=analyze_and_save_video, **button_style)
btn_save_video.pack(pady=10)

btn_camera = tk.Button(button_frame, text="CAMERA", command=lambda: show_camera(display_label), **button_style)
btn_camera.pack(pady=10)

display_label = tk.Label(root, bg="#d9d9d9", width=800, height=600)
display_label.pack(side=tk.RIGHT, padx=20, pady=20)

# Na tę wersję z grubszym paskiem:
progress_var = tk.DoubleVar()
style = ttk.Style()
style.configure("Custom.Horizontal.TProgressbar", thickness=30)  # Ustaw grubość paska
progress_bar = ttk.Progressbar(
    button_frame,
    variable=progress_var,
    maximum=100,
    length=200,
    style="Custom.Horizontal.TProgressbar"
)
progress_bar.pack(pady=20)

root.mainloop()