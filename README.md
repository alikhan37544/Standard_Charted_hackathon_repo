# 🚀 Standard Chartered Hackathon Project  

An AI-powered **virtual branch manager** that interacts with customers through **video-based conversations**, guiding them through **loan applications, document submission, and eligibility checks**. This system provides a seamless, branch-like banking experience with **facial verification, speech processing, document classification, and AI-driven loan decisioning**.  
This repository contains multiple components developed for the Standard Chartered hackathon, including:  
- **Aadhaar Card Recognition & Verification System:** Verifies Aadhaar card authenticity using OCR and data validation.  
- **Picture Reader:** Reads and processes images for document verification.  
- **Verification Transcription:** Handles text extraction and validation.  
- **Fintech Chatbot:** Financial assistant chatbot with NLP capabilities.  

---

## **🔹 Key Features**
### 🎥 **Real-Time Facial Verification**
- Captures images periodically (every **10 seconds**) to verify identity consistency.  
- Ensures continuous applicant validation throughout the process.  

### 🔊 **Dual-Mode Interaction**
- Simultaneous **face and speech processing** reduces manual effort.  
- Provides a more interactive and seamless experience.  

### 📄 **Intelligent Document Parsing**
- When tested with an **external Aadhaar sample**, the CNN successfully identified the **Aadhaar number and name**.  
- Extracts and validates key details using OCR and rule-based verification.  

### ⚙️ **Multi-Threaded Processing**
- **Parallel processing** handles face verification, transcription, and scoring concurrently.  
- Improves efficiency and reduces processing time.  

### 🚀 **Rule-Based Loan Eligibility Scoring**
- Uses a **rule-based system** to evaluate eligibility factors.  
- ML-based scoring planned for future implementation to enhance accuracy.  

---

## 🛠️ **Technologies Used**
- Python  
- OpenCV  
- Flask  
- TensorFlow/Keras  
- PaddleOCR  
- NLP Libraries  

---

## **🔹 Workflow Diagram**
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
```

### **3️⃣ Run the AI Chatbot**
```bash
python fintech_chatbot/main.py
```

---

## 📂 **File Structure**
```
📂 Standard_Charted_hackathon_repo  
 ├── 📂 Aadhaar_Card_Recognition/       # OCR & Aadhaar validation  
 ├── 📂 Picture_reader/                  # Image processing for document verification  
 ├── 📂 Verification_transcription/     # Speech-to-text for user verification  
 ├── 📂 fintech_chatbot/                # AI chatbot guiding users through banking processes  
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

### **4️⃣ Loan Eligibility Check**
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

4. **Loan Eligibility Evaluation:**  
   - Rule-based eligibility check provides instant feedback.  

---

## 🚀 **Future Enhancements**
- ✅ **Real-time Loan Offer Suggestions**  
- ✅ **Integration with Core Banking Systems**  
- ✅ **Automated Loan Document E-Signing**  
- ✅ **ML-Powered Eligibility Scoring** for more accurate predictions  
