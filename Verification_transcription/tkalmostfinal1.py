import cv2
import time
import os
import threading
import tkinter as tk
from PIL import Image, ImageTk
from deepface import DeepFace
import speech_recognition as sr

class FacialVerificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Facial Verification with Speech Recognition")
        self.root.geometry("800x600")

        # Parameters
        self.capture_interval = 10  # Capture every 10 seconds
        self.last_capture_time = 0
        self.monitoring = False
        self.aadhaar_mode = False
        self.captured_images = []
        self.speech_recorder = SpeechToTextRecorder()  # Initialize speech recorder

        # File paths
        self.reference_image = "initial_reference.jpg"
        self.aadhar_folder = "aadhaar_captures"
        self.aadhar_image = os.path.join(self.aadhar_folder, "aadhar_card.jpg")
        os.makedirs(self.aadhar_folder, exist_ok=True)

        # OpenCV Camera
        self.cap = cv2.VideoCapture(0)

        # UI Elements
        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()

        self.status_label = tk.Label(self.root, text="Press 'Capture Reference' to start", font=("Arial", 12))
        self.status_label.pack()

        # Buttons
        self.capture_btn = tk.Button(self.root, text="Capture Reference", command=self.capture_reference)
        self.capture_btn.pack()

        self.start_btn = tk.Button(self.root, text="Start Monitoring", command=self.start_monitoring)
        self.start_btn.pack()

        self.aadhaar_btn = tk.Button(self.root, text="Capture Aadhaar", command=self.prepare_aadhar_capture)
        self.aadhaar_btn.pack()

        self.exit_btn = tk.Button(self.root, text="Exit & Verify", command=self.exit_and_verify)
        self.exit_btn.pack()

        # Start the camera feed loop
        self.update_camera_feed()

    def update_camera_feed(self):
        """Updates the camera feed in the Tkinter window."""
        ret, frame = self.cap.read()
        if ret:
            if self.aadhaar_mode:
                frame = self.draw_aadhar_box(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk  # Keep reference

        self.root.after(10, self.update_camera_feed)

    def capture_reference(self):
        """Captures the initial reference image."""
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite(self.reference_image, frame)
            self.status_label.config(text="✅ Reference Image Captured!")
            print(f"✅ Reference Image saved as {self.reference_image}")

    def start_monitoring(self):
        """Starts facial monitoring and speech transcription."""
        if not os.path.exists(self.reference_image):
            self.status_label.config(text="⚠️ Capture Reference First!")
            return

        self.monitoring = True
        self.status_label.config(text="🎥 Monitoring Started...")
        threading.Thread(target=self.monitor_faces, daemon=True).start()

        # Start speech recognition in the background
        self.speech_recorder.start()

    def draw_aadhar_box(self, frame):
        """Draws a guide box for Aadhaar card alignment."""
        h, w, _ = frame.shape
        box_width, box_height = 450, 300
        x1, y1 = (w - box_width) // 2, (h - box_height) * 3 // 4
        x2, y2 = x1 + box_width, y1 + box_height

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Align Aadhaar Card Here", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    def prepare_aadhar_capture(self):
        """Enters Aadhaar capture mode and waits 5 seconds before capturing."""
        self.aadhaar_mode = True
        self.status_label.config(text="🆔 Show Aadhaar card in the box. Capturing in 5s...")
        self.root.update()

        threading.Thread(target=self.capture_aadhar, daemon=True).start()

    def capture_aadhar(self):
        """Captures Aadhaar card after 5 seconds."""
        time.sleep(5)

        ret, frame = self.cap.read()
        if ret:
            frame_without_box = frame.copy()
            self.aadhaar_mode = False
            self.status_label.config(text="✅ Aadhaar Captured!")
            cv2.imwrite(self.aadhar_image, frame_without_box)
            print(f"✅ Aadhaar card captured at {self.aadhar_image}")

    def monitor_faces(self):
        """Continuously captures images every 10s and verifies identity."""
        start_time = time.time()

        while self.monitoring:
            elapsed_time = time.time() - start_time

            if elapsed_time - self.last_capture_time >= self.capture_interval and not self.aadhaar_mode:
                image_name = f"capture_{int(time.time())}.jpg"
                ret, frame = self.cap.read()
                if ret:
                    cv2.imwrite(image_name, frame)
                    self.captured_images.append(image_name)
                    self.last_capture_time = elapsed_time
                    self.status_label.config(text=f"📸 Captured: {image_name}")
                    print(f"📸 Captured: {image_name}")

            time.sleep(1)

    def verify_identity(self):
        """Verifies all captured images against the initial reference image."""
        if not self.captured_images:
            self.status_label.config(text="⚠️ No images captured!")
            return

        for img in self.captured_images:
            try:
                result = DeepFace.verify(self.reference_image, img, model_name="Facenet")
                if not result["verified"]:
                    self.status_label.config(text=f"❌ Face Mismatch Detected!")
                    print(f"❌ Face mismatch detected at {img}!")
                    return
            except Exception as e:
                print(f"⚠️ Error verifying {img}: {e}")

        self.status_label.config(text="✅ Same Person Detected Throughout!")
        print("✅ Same person detected throughout the session!")

    def exit_and_verify(self):
        """Stops monitoring, verifies identity, and stops speech recognition."""
        self.monitoring = False
        self.cap.release()
        self.speech_recorder.stop()
        self.verify_identity()
        self.root.quit()

class SpeechToTextRecorder:
    def __init__(self, output_file="transcription.txt"):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.output_file = output_file
        self.running = True

        with open(self.output_file, "w") as f:
            f.write("Speech Transcription Log\n---------------------------\n")

    def listen_and_transcribe(self):
        """Continuously listens for speech and transcribes it."""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("🎤 Listening for speech...")

        while self.running:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                print(f"📝 Transcribed: {text}")

                with open(self.output_file, "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {text}\n")

            except sr.UnknownValueError:
                print("⚠️ Could not understand audio")
            except sr.RequestError as e:
                print(f"❌ Google Speech API error: {e}")
            except Exception as e:
                print(f"⚠️ Error: {e}")

    def start(self):
        """Starts speech transcription in a separate thread."""
        thread = threading.Thread(target=self.listen_and_transcribe, daemon=True)
        thread.start()

    def stop(self):
        """Stops the transcription process."""
        self.running = False
        print("🛑 Stopping speech transcription...")

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialVerificationApp(root)
    root.mainloop()
