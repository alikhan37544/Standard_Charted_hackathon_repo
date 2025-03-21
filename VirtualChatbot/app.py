import streamlit as st
import imageio
import numpy as np
import skimage.transform
import os
from skimage import img_as_ubyte
from modules.demo import load_checkpoints, make_animation
import tempfile

# 🌟 Ensure upload directory exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🎨 Streamlit UI - Modern Design
st.set_page_config(page_title="AI Virtual Assistant", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background-color: #f4f4f4;}
    .main-title {text-align: center; font-size: 2em; font-weight: bold; color: #333;}
    .sub-title {text-align: center; font-size: 1.2em; color: #666;}
    .stButton>button {width: 100%; font-size: 16px;}
    </style>
    """,
    unsafe_allow_html=True
)

# 🏠 **Header**
st.markdown('<p class="main-title">🎭 AI Virtual Assistant Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload an image and a driving video to create an animated assistant!</p>', unsafe_allow_html=True)

# 🔍 **Sidebar: File Uploads**
st.sidebar.header("📂 Upload Files")
image_file = st.sidebar.file_uploader("🖼 Upload an Image", type=["png", "jpg", "jpeg"])
video_file = st.sidebar.file_uploader("🎥 Upload a Driving Video", type=["mp4", "mov"])

# ⚙️ **Animation Settings**
st.sidebar.header("⚙️ Settings")
relative_motion = st.sidebar.checkbox("Relative Keypoint Displacement", value=True)
adapt_movement = st.sidebar.checkbox("Adapt Movement Scale", value=True)

if image_file and video_file:
    st.sidebar.success("✅ Files Uploaded Successfully!")
    
    # Show uploaded images & videos
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_file, caption="🖼 Uploaded Image", use_column_width=True)
    with col2:
        st.video(video_file, format="video/mp4", start_time=0)

    # 🏗 **Processing Animation**
    if st.sidebar.button("🚀 Generate Animation"):
        with st.spinner("✨ Creating Animation... Please Wait..."):
            # Save uploaded files
            image_path = os.path.join(UPLOAD_FOLDER, "input_image.png")
            video_path = os.path.join(UPLOAD_FOLDER, "input_video.mp4")
            
            with open(image_path, "wb") as f:
                f.write(image_file.read())
            with open(video_path, "wb") as f:
                f.write(video_file.read())
            
            # 🔥 Load the AI Model
            generator, kp_detector = load_checkpoints(
                config_path="config/vox-256.yaml",
                checkpoint_path="checkpoints/vox-cpk.pth",
                cpu=True  # Force CPU mode
            )

            # 📥 Process Image & Video
            image = imageio.imread(image_path)
            reader = imageio.get_reader(video_path, mode="I", format="FFMPEG")
            fps = reader.get_meta_data()["fps"]
            driving_video = [frame for frame in reader]

            predictions = make_animation(
                skimage.transform.resize(image, (256, 256)),
                [skimage.transform.resize(frame, (256, 256)) for frame in driving_video],
                generator, kp_detector,
                relative=relative_motion, adapt_movement_scale=adapt_movement
            )

            # 🎞 Save & Display the Output Video
            output_path = os.path.join(UPLOAD_FOLDER, "output.mp4")
            imageio.mimsave(output_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)

            st.success("✅ Animation Generated Successfully!")
            st.video(output_path, format="video/mp4")

            # 📥 Download Button
            with open(output_path, "rb") as file:
                st.download_button(label="📥 Download Animated Video", data=file, file_name="animated_output.mp4", mime="video/mp4")

