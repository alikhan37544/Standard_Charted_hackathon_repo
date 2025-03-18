from data_augmentation import run_augmentation
from model_training import train_model
from ocr_extraction import process_aadhaar_card

def main():
    run_augmentation()
    train_model()
    print(process_aadhaar_card("sample_aadhaar.jpg"))

if __name__ == "__main__":
    main()
