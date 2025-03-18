import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Try this for Linux
# Or alternatively:
# os.environ['OPENCV_VIDEOIO_DEBUG'] = '1'  # Enable debug output for video capture

import cv2 
import numpy as np
import mtcnn
from architecture import *
import os
import sys  # Ensure sys is imported
import time
import pickle
import tensorflow as tf
from scipy.spatial.distance import cosine
import argparse
from collections import OrderedDict, deque
import dlib
from imutils import face_utils
import mediapipe as mp
import logging

# Suppress MediaPipe logging output
logging.getLogger("mediapipe").setLevel(logging.ERROR)
os.environ["GLOG_minloglevel"] = "2"  # Suppress MediaPipe C++ logging
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'  # For Windows
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # For Linux with Qt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Face Detection and Recognition')
parser.add_argument('--encodings', default='encodings/encodings.pkl', help='Path to encodings file')
parser.add_argument('--model', default='facenet_keras_weights.h5', help='Path to facenet model')
parser.add_argument('--confidence', type=float, default=0.98, help='Minimum confidence for face detection')
parser.add_argument('--recognition-threshold', type=float, default=0.4, help='Recognition cosine distance threshold')
parser.add_argument('--source', default=0, help='Video source (0 for webcam, or path to video file)')
parser.add_argument('--display-size', default='1280x720', help='Display resolution WxH')
parser.add_argument('--blur-background', action='store_true', help='Blur background around faces')
parser.add_argument('--enable-anti-spoofing', action='store_true', help='Enable liveness detection')
args = parser.parse_args()

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Constants
confidence_t = args.confidence
recognition_t = args.recognition_threshold
required_size = (160, 160)
width, height = map(int, args.display_size.split('x'))

from sklearn.preprocessing import Normalizer
l2_normalizer = Normalizer('l2')

# Initialize MediaPipe Face Mesh for better eye tracking
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Initialize with silent output
try:
    # Redirect both stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    sys.stderr = devnull
    
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Restore stdout and stderr
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    devnull.close()
    print("MediaPipe Face Mesh initialized successfully")
    
except Exception as e:
    # Make sure to restore stdout/stderr even if there's an error
    if 'old_stdout' in locals():
        sys.stdout = old_stdout
    if 'old_stderr' in locals():
        sys.stderr = old_stderr
    if 'devnull' in locals() and not devnull.closed:
        devnull.close()
    print(f"Error initializing MediaPipe Face Mesh: {e}")
    # Still define face_mesh to avoid NameError later
    face_mesh = None

# MediaPipe eye landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Blink detection parameters
EAR_THRESHOLD_RATIO = 0.75
MIN_BLINK_FRAMES = 2
BLINK_COOLDOWN = 0.2
ENABLE_ANTI_SPOOFING = args.enable_anti_spoofing

# State variables for improved blink detection
frames_below_threshold = 0
blink_in_progress = False
cooldown_active = False
last_blink_time = time.time()
BLINK_COUNTER = 0
ear_history = deque(maxlen=60)
baseline_ear = None
calibration_complete = False
calibration_frames = 0

# Load facial landmark predictor for eye blink detection (anti-spoofing)
predictor_path = "shape_predictor_68_face_landmarks.dat"
if args.enable_anti_spoofing and os.path.exists(predictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    ENABLE_ANTI_SPOOFING = True
    print("Anti-spoofing module enabled")
else:
    ENABLE_ANTI_SPOOFING = False
    if args.enable_anti_spoofing:
        print(f"Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print(f"Extract it to {predictor_path}")

# Improved eye aspect ratio function for blink detection
def eye_aspect_ratio(landmarks, eye_indices):
    # Get the eye landmark coordinates
    points = [landmarks[idx] for idx in eye_indices]
    
    # Calculate the vertical distances
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    
    # Calculate the horizontal distance
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    
    # Calculate EAR
    ear = (A + B) / (2.0 * C) if C > 0 else 0
    return ear

# Constants for blink detection
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3
BLINK_COUNTER = 0

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def detect(img, detector, encoder, encoding_dict):
    global BLINK_COUNTER, baseline_ear, calibration_complete, calibration_frames
    global frames_below_threshold, blink_in_progress, cooldown_active, last_blink_time, ear_history
    global fps  # Add fps as global variable
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # First use MTCNN to detect faces (faster initial detection)
    results = detector.detect_faces(img_rgb)
    
    # Early return if no faces
    if not results:
        fps_text = f"FPS: {int(fps)}" if fps > 0 else "FPS: calculating..."  # Modified
        result_img = img.copy()
        cv2.putText(result_img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result_img, "People: 0", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return result_img
    
    # Create a copy for background blur if needed
    if args.blur_background:
        blurred_img = cv2.GaussianBlur(img, (25, 25), 0)
    
    # Final result image
    result_img = img.copy() if not args.blur_background else blurred_img

    # Keep track of recognized faces
    recognized_faces = []
    
    # Process face with MediaPipe for more accurate landmarks if anti-spoofing is enabled
    is_live = True
    if ENABLE_ANTI_SPOOFING and face_mesh is not None:
        # For performance, mark image as not writeable
        img_rgb.flags.writeable = False
        mp_results = face_mesh.process(img_rgb)
        img_rgb.flags.writeable = True
        
        if mp_results and mp_results.multi_face_landmarks:
            face_landmarks = mp_results.multi_face_landmarks[0]  # Use first detected face
            
            # Extract coordinates for easier processing
            landmarks = {}
            for idx, landmark in enumerate(face_landmarks.landmark):
                landmarks[idx] = (int(landmark.x * img.shape[1]), 
                                int(landmark.y * img.shape[0]))
            
            # Calculate EAR for each eye and use average
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear_avg = (left_ear + right_ear) / 2.0
            
            # Store in history
            ear_history.append(ear_avg)
            
            # Calibration phase for personalized threshold
            if not calibration_complete:
                calibration_frames += 1
                
                # After collecting enough samples, calculate baseline
                if calibration_frames >= 30:  # 1 second at 30fps
                    sorted_ears = sorted(ear_history)
                    
                    # Use the upper 70% of values for baseline
                    baseline_ear = np.mean(sorted_ears[int(len(sorted_ears)*0.3):])
                    calibration_complete = True
                    threshold = baseline_ear * EAR_THRESHOLD_RATIO
                    
                    print(f"Blink calibration complete. Baseline EAR: {baseline_ear:.4f}, Threshold: {threshold:.4f}")
            
            # Only proceed with blink detection after calibration
            elif calibration_complete:
                # Calculate dynamic threshold based on baseline
                threshold = baseline_ear * EAR_THRESHOLD_RATIO
                
                current_time = time.time()
                
                if not cooldown_active:
                    # Check for blink
                    if ear_avg < threshold and not blink_in_progress:
                        frames_below_threshold += 1
                        
                        # Confirm blink after minimum frames
                        if frames_below_threshold >= MIN_BLINK_FRAMES:
                            blink_in_progress = True

                    # Check if eyes opened again after blink
                    elif ear_avg > threshold * 1.1 and blink_in_progress:
                        BLINK_COUNTER += 1
                        frames_below_threshold = 0
                        blink_in_progress = False
                        cooldown_active = True
                        last_blink_time = current_time
                        print(f"Blink detected! Count: {BLINK_COUNTER}")
                        
                    # Reset counter if not enough consecutive frames
                    elif ear_avg > threshold and frames_below_threshold > 0:
                        frames_below_threshold = 0
                else:
                    # Handle cooldown period
                    if current_time - last_blink_time > BLINK_COOLDOWN:
                        cooldown_active = False
                
                # If we haven't seen a blink recently, it might be a photo
                is_live = BLINK_COUNTER > 0 and (current_time - last_blink_time < 5.0)
                
                # Draw eye landmarks for visualization
                if is_live:
                    # Left eye
                    for idx in LEFT_EYE:
                        cv2.circle(result_img, landmarks[idx], 2, (0, 255, 0), -1)
                    
                    # Right eye
                    for idx in RIGHT_EYE:
                        cv2.circle(result_img, landmarks[idx], 2, (0, 255, 0), -1)
    
    # Prepare batch processing for face encodings
    faces_to_encode = []
    face_locations = []
    
    # Collect all faces for batch processing
    for res in results:
        if res['confidence'] < confidence_t:
            continue
            
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        
        if face.size == 0:
            continue
        
        # Normalize and resize
        face_norm = normalize(face)
        face_resized = cv2.resize(face_norm, required_size)
        
        # Add to batch
        faces_to_encode.append(face_resized)
        face_locations.append((face, pt_1, pt_2, res))
    
    # Get encodings for all faces at once (if any faces were detected)
    if faces_to_encode:
        face_encodings = encoder.predict(np.array(faces_to_encode))
        face_encodings = l2_normalizer.transform(face_encodings)
    
    # Now process each face with its encoding
    for i, (face, pt_1, pt_2, res) in enumerate(face_locations):
        # Get face encoding and match
        encode = face_encodings[i]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        # Calculate face region for unblurring
        if args.blur_background:
            # Copy the unblurred face region back to the result image
            result_img[pt_1[1]:pt_2[1], pt_1[0]:pt_2[0]] = img[pt_1[1]:pt_2[1], pt_1[0]:pt_2[0]]
        
        # Draw results
        text_color = (255, 255, 255)  # white text
        if name == 'unknown':
            bbox_color = (0, 0, 255)  # red box for unknown
            cv2.rectangle(result_img, pt_1, pt_2, bbox_color, 2)
            
            # Create filled background for text
            text = name
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(result_img, 
                         (pt_1[0], pt_1[1] - text_size[1] - 10), 
                         (pt_1[0] + text_size[0], pt_1[1]), 
                         bbox_color, -1)
            cv2.putText(result_img, text, (pt_1[0], pt_1[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        else:
            bbox_color = (0, 255, 0) if is_live else (0, 165, 255)  # green if live, orange if potential spoof
            cv2.rectangle(result_img, pt_1, pt_2, bbox_color, 2)
            
            # Create filled background for text
            conf_text = f"{name} ({distance:.2f})"
            text_size, _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(result_img, 
                         (pt_1[0], pt_1[1] - text_size[1] - 10), 
                         (pt_1[0] + text_size[0], pt_1[1]), 
                         bbox_color, -1)
            cv2.putText(result_img, conf_text, (pt_1[0], pt_1[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            
            # Add liveness indicator
            if ENABLE_ANTI_SPOOFING:
                live_text = "LIVE" if is_live else "SPOOF?"
                live_color = (0, 255, 0) if is_live else (0, 0, 255)
                
                # Place liveness indicator below the face
                cv2.putText(result_img, live_text, 
                           (pt_1[0], pt_2[1] + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, live_color, 2)
                
                # Show blink count
                cv2.putText(result_img, f"Blinks: {BLINK_COUNTER}", 
                           (pt_1[0], pt_2[1] + 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Track the recognized face
            if is_live:
                recognized_faces.append(name)

    # Display stats - fixed fps reference
    fps_text = f"FPS: {int(fps)}" if fps > 0 else "FPS: calculating..."
    cv2.putText(result_img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Show recognized people count
    recognized_set = set(recognized_faces)
    cv2.putText(result_img, f"People: {len(recognized_set)}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Return the result image
    return result_img

if __name__ == "__main__":
    # Load models and encodings
    print("Loading face recognition model...")
    face_encoder = InceptionResNetV2()
    face_encoder.load_weights(args.model)
    
    print(f"Loading encodings from {args.encodings}...")
    encoding_dict = load_pickle(args.encodings)
    print(f"Loaded {len(encoding_dict)} face encodings")
    
    print("Initializing face detector...")
    face_detector = mtcnn.MTCNN(device='CPU:0')
    
    print(f"Opening video source: {args.source}")
    try:
        if isinstance(args.source, int) or (isinstance(args.source, str) and args.source.isdigit()):
            cap = cv2.VideoCapture(int(args.source))
        else:
            cap = cv2.VideoCapture(args.source)
            
        if not cap.isOpened():
            raise Exception(f"Could not open video source: {args.source}")
        
        # Crucial for proper camera initialization
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your camera connection or video file path.")
        exit(1)
    
    # For FPS calculation - initialize globally
    global fps
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    # Create a named window
    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Detection", width, height)
    
    print("Starting video stream, press 'q' to quit or 'r' to recalibrate...")
    
    while True:
        # Read frame with retry logic
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, retrying...")
            # Try to reconnect to camera
            cap.release()
            cap = cv2.VideoCapture(int(args.source) if isinstance(args.source, int) or args.source.isdigit() else args.source)
            if not cap.isOpened():
                print("Could not reconnect to camera. Exiting.")
                break
            continue
            
        # Resize frame if needed
        if frame.shape[1] > width or frame.shape[0] > height:
            frame = cv2.resize(frame, (width, height))
            
        # Process frame
        result_frame = detect(frame, face_detector, face_encoder, encoding_dict)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:  # Update FPS every second
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.1f}")
            frame_count = 0
            start_time = time.time()
            
        # Display the processed frame
        cv2.imshow("Face Detection", result_frame)
        
        # Exit on 'q' press, reset calibration on 'r' press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Recalibrating blink detection...")
            calibration_complete = False
            calibration_frames = 0
            ear_history.clear()
            BLINK_COUNTER = 0
            
    # Clean up
    cap.release()
    cv2.destroyAllWindows()




