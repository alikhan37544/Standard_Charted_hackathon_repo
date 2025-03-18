# 🚀 Standard Chartered Hackathon Project  

An AI-powered **virtual branch manager** that interacts with customers through **video-based conversations**, guiding them through **loan applications, document submission, and eligibility checks**. This system provides a seamless, branch-like banking experience with **facial verification, speech processing, document classification, and AI-driven loan decisioning**.
This repository contains multiple components developed for the Standard Chartered hackathon, including:  
- **Aadhaar Card Recognition & Verification System:** Verifies Aadhaar card authenticity using OCR and data validation.  
- **Picture Reader:** Reads and processes images, likely for document verification.  
- **Verification Transcription:** Handles text extraction and validation.  
- **Fintech Chatbot:** Financial assistant chatbot with NLP capabilities.

---

## **🔹 Key Features**
✅ **Virtual AI Branch Manager** – Chatbot guides users through the loan process.  
✅ **Facial & Speech Verification** – Ensures identity consistency.  
✅ **OCR-Based Document Validation** – Extracts data from Aadhaar, PAN, and income proof.  
✅ **Automated Loan Eligibility Check** – Rule-based AI determines approval.  
✅ **Multi-Language Support** – Enhances accessibility.  

---

## ⚙️ **Technologies Used**
- Python  
- OpenCV  
- Flask  
- TensorFlow/Keras  
- NLP Libraries  

---

## **🔹 Workflow Diagram**

![_- visual selection (1)](https://github.com/user-attachments/assets/ebb2cd81-1ef3-4dd9-9e19-1f7e3f304429)

---

## **🔹 Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/alikhan37544/Standard_Charted_hackathon_repo.git
cd Standard_Charted_hackathon_repo
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the AI Chatbot**
```bash
python fintech_chatbot/main.py
```

---

## **🔹 File Structure**
```
📂 Standard_Charted_hackathon_repo  
 ├── 📂 Aadhaar Card Recognition & Verification System/  # OCR & Aadhaar validation  
 ├── 📂 Picture_reader (Copy)/  # Image processing for document recognition  
 ├── 📂 Verification_transcription/  # Speech-to-text for user verification  
 ├── 📂 fintech_chatbot (Copy)/  # AI chatbot guiding users through banking processes  
 ├── 📄 README.md  # Project Overview  
 ├── 📄 requirements.txt  # Dependencies  
```

---

## **🔹 How It Works**
### **1️⃣ Chatbot Interaction**
- User initiates conversation via chatbot (`fintech_chatbot/main.py`).  
- AI **guides them through the loan application process**.  

### **2️⃣ Facial & Speech Verification**
- `app.py` verifies user **identity continuity**.  
- Ensures the same applicant **submits documents & applies for loans**.  

### **3️⃣ Document Validation**
- **Aadhaar Card Recognition System** processes **Aadhaar, PAN, and income proof**.  
- Uses **OCR & AI-based classification** to extract data.  

### **4️⃣ Loan Eligibility Check**
- `loan_eligibility_checker.py` applies **rule-based AI** to evaluate:  
  - Age, income, employment type, credit score.  
  - Provides **instant loan decision**: ✅ Approved | ❌ Rejected | 🔄 More Info Needed.  

---

## **🔹 Requirements**
Below is an **accurate** `requirements.txt`, based on your uploaded files.

### **📌 `requirements.txt`**
```
ollama
opencv-python
numpy
pandas
scikit-learn
tensorflow
paddleocr
pytesseract
deepface
speechrecognition
pillow
tkinter
```

---

## **🔹 Future Enhancements**
🚀 **Real-time Loan Offer Suggestions**  
🚀 **Integration with Core Banking Systems**  
🚀 **Automated Loan Document E-Signing**  
