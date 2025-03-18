# 🚀 Standard Chartered Hackathon Project  

An AI-powered **virtual branch manager** that interacts with customers through **video-based conversations**, guiding them through **loan applications, document submission, and eligibility checks**. This system provides a seamless, branch-like banking experience with **facial verification, speech processing, document classification, and AI-driven loan decisioning**.  
This repository contains multiple components developed for the Standard Chartered hackathon, including:  
- **Aadhaar Card Recognition & Verification System:** Verifies Aadhaar card authenticity using OCR and data validation.  
- **Picture Reader:** Reads and processes images for document verification.  
- **Verification Transcription:** Handles text extraction and validation.  
- **Fintech Chatbot:** Financial assistant chatbot with NLP capabilities.  
- **Ollama Image Analysis Module:** Uses **Ollama API with the LLaVA model** to analyze Aadhaar images and generate textual descriptions.  
- **Face Detection & Recognition Module:** Leverages **MTCNN and custom CNN architecture** for facial verification.  

---

## 🔹 **Key Features**

### 🎥 **Real-Time Facial Verification**
- Captures images periodically (every **10 seconds**) to verify identity consistency.  
- Ensures continuous applicant validation throughout the process.  

### 🔊 **Dual-Mode Interaction**
- Simultaneous **face and speech processing** reduces manual effort.  
- Provides a more interactive and seamless experience.  

### 📄 **Intelligent Document Parsing**
- When tested with an **external Aadhaar sample**, the CNN successfully identified the **Aadhaar number and name**.  
- Extracts and validates key details using OCR and rule-based verification.  

### 🔍 **Ollama Image Analysis**
- Uses the **Ollama API with the LLaVA model** to analyze Aadhaar images.  
- Generates **textual descriptions** of the Aadhaar card, identifying key details such as:  
  - **Profile picture**  
  - **Name, DOB, and gender**  
  - **Unique ID number and QR code**  
  - **Multi-language text and government emblem**  

### 🔥 **Face Detection & Recognition**
- **MTCNN-based face detection** for efficient multi-scale face recognition.  
- Custom **CNN architecture** for improved accuracy in facial verification.  
- **Multi-threaded processing** handles face recognition concurrently with speech transcription.  

### 🚀 **Rule-Based Loan Eligibility Scoring**
- Uses a **rule-based system** to evaluate eligibility factors.  
- ML-based scoring planned for future implementation to enhance accuracy.  

---

## 🛠️ **Technologies Used**
- **Python**  
- **OpenCV**  
- **Flask**  
- **TensorFlow/Keras**  
- **PaddleOCR**  
- **NLP Libraries**  
- **Ollama API**  
- **LLaVA Model**  
- **MTCNN** for face detection  
- **Custom CNN architecture**  

---

## 🔹 **Workflow Diagram**
![Workflow](https://github.com/user-attachments/assets/ebb2cd81-1ef3-4dd9-9e19-1f7e3f304429)

---

## ✅ **Installation & Setup**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/alikhan37544/Standard_Charted_hackathon_repo.git
cd Standard_Charted_hackathon_repo
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
pip install ollama
```

### **3️⃣ Run the AI Chatbot**
```bash
python fintech_chatbot/main.py
```

### **4️⃣ Run Ollama Image Analysis Module**
```bash
python image_analysis.py --image_path /path/to/aadhaar_image.jpg
```

### **5️⃣ Run MTCNN Face Detection & Recognition**
```bash
python detect.py
```

---

## 📂 **File Structure**
```
📂 Standard_Charted_hackathon_repo  
 ├── 📂 Aadhaar_Card_Recognition/       # OCR & Aadhaar validation  
 ├── 📂 Picture_reader/                  # Image processing for document verification  
 ├── 📂 Verification_transcription/     # Speech-to-text for user verification  
 ├── 📂 fintech_chatbot/                # AI chatbot guiding users through banking processes  
 ├── 📂 ollama_image_analysis/          # Image analysis module with Ollama API  
 ├── 📂 face_recognition/               # MTCNN-based face detection & CNN recognition   
 ├── 📄 README.md                        # Project Overview  
 ├── 📄 requirements.txt                 # Dependencies  
```

---

## 🚀 **How It Works**

### **1️⃣ Chatbot Interaction**
- User initiates conversation via chatbot (`fintech_chatbot/main.py`).  
- AI **guides them through the loan application process**.  

### **2️⃣ Facial & Speech Verification**
- `app.py` verifies user **identity continuity**.  
- Ensures the same applicant **submits documents & applies for loans**.  

### **3️⃣ Document Validation**
- **Aadhaar Card Recognition System** processes **Aadhaar, PAN, and income proof**.  
- Uses **OCR & AI-based classification** to extract data.  

### **4️⃣ Ollama Image Analysis**
- The **Ollama API with LLaVA model** processes the Aadhaar image.  
- Generates a detailed textual description of the card, including:  
  - **Personal details**  
  - **Document format**  
  - **QR code and multi-language content**  

### **5️⃣ Face Detection & Recognition**
- `detect.py` uses **MTCNN for face detection**.  
- Recognizes and verifies faces using the **custom CNN architecture**.  

### **6️⃣ Loan Eligibility Check**
- `loan_eligibility_checker.py` applies **rule-based AI** to evaluate:  
  - Age, income, employment type, and credit score.  
  - Provides **instant loan decision**: ✅ Approved | ❌ Rejected | 🔄 More Info Needed.  

---

## 🚦 **User Flow**

1. **Start the AI Loan Assistant:**  
   - Launches the **Fintech Chatbot**.  
   - Walks the user through the loan application steps.  

2. **Facial & Speech Verification:**  
   - Ensures continuous identity verification during the interaction.  

3. **Document Upload & Parsing:**  
   - Extracts and validates Aadhaar and PAN details using OCR.  

4. **Ollama Image Analysis:**  
   - Adds a second layer of verification with **textual descriptions** of Aadhaar images.  

5. **Face Recognition:**  
   - Verifies the applicant's face consistency throughout the process.  

6. **Loan Eligibility Evaluation:**  
   - Rule-based eligibility check provides instant feedback.  

---

## 🚀 **Future Enhancements**
- ✅ **Real-time Loan Offer Suggestions**  
- ✅ **Integration with Core Banking Systems**  
- ✅ **Automated Loan Document E-Signing**  
- ✅ **ML-Powered Eligibility Scoring** for more accurate predictions  
- ✅ **Document Authenticity Detection** using advanced AI  
- ✅ **Fraud Detection Module** leveraging behavioral analytics  

---

## 🛠️ **Contributors**
- **Ali Khan** – 
- **Pragatish A M** – Transcription & Facial Verification  
- **Shrinjita Paul** – Loan Eligibility Model & Aadhar Verification CNN
- **Shreeharini S** –
- **Koushik Babu** -
