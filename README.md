
#### **1️⃣ Installation**  
Before running the project, install all dependencies:  
```bash
pip install -r requirements.txt
```

#### **2️⃣ Execution**  
Run the **main script** to execute the full pipeline:  
```bash
python main.py
```

#### **3️⃣ Customization (Changes to Make Before Running on Your System)**  

✅ **Modify File Paths**  
Edit `config.py` to set **correct file paths** for images and dataset directories:  
```python
# Update these paths to your local dataset locations
input_dir_aadhar = "/path/to/aadhar/images"
input_dir_control = "/path/to/non-aadhar/images"
output_dir_aadhar = "/path/to/save/augmented/aadhar"
output_dir_control = "/path/to/save/augmented/non-aadhar"

base_dir = "/path/to/dataset"
```

✅ **Tesseract Path (Windows Only)**  
If using Windows, update `ocr_extraction.py`:  
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

✅ **Adjust Model Parameters**  
Modify **batch size, image size, and epochs** in `config.py` to fit your system's GPU/CPU:  
```python
img_size = (224, 224)
batch_size = 16  # Reduce if memory is limited
epochs = 10
```

To install dependencies:  
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
