import os
import subprocess

# Required document list
REQUIRED_DOCUMENTS = ["aadhaar.jpg", "pan.jpg", "income_proof.jpg"]

def check_documents():
    """Checks if all required documents are uploaded."""
    missing_docs = [doc for doc in REQUIRED_DOCUMENTS if not os.path.exists(doc)]
    
    if missing_docs:
        return False, f"Missing documents: {', '.join(missing_docs)}"
    return True, "All required documents are present."

def main():
    docs_verified, message = check_documents()
    print(message)
    
    if docs_verified:
        print("Redirecting to loan eligibility check...")
        subprocess.run(["python", "loan_eligibility_checker.py"])
    else:
        print("Please upload all required documents before proceeding.")

if __name__ == "__main__":
    main()
