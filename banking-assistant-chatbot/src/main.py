from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize the Ollama LLM
llm = Ollama(model="llama3.2")

# Define prompt templates for user interaction
prompt_templates = {
    "greeting": "Hello! How can I assist you today? You can choose to make a new bank account, create a new savings account, or open a loan.",
    "account_creation": "You have chosen to make a new bank account. Please provide your details.",
    "savings_account": "You have chosen to create a new savings account. Please provide your details.",
    "loan": "You have chosen to open a loan. Please provide your details."
}

def get_user_choice():
    print(prompt_templates["greeting"])
    user_input = input("Your choice: ").strip().lower()
    
    if "bank account" in user_input:
        return "account_creation"
    elif "savings account" in user_input:
        return "savings_account"
    elif "loan" in user_input:
        return "loan"
    else:
        print("I'm sorry, I didn't understand that. Please choose one of the options.")
        return get_user_choice()

def main():
    user_choice = get_user_choice()
    
    if user_choice == "account_creation":
        response = llm(prompt_templates["account_creation"])
    elif user_choice == "savings_account":
        response = llm(prompt_templates["savings_account"])
    elif user_choice == "loan":
        response = llm(prompt_templates["loan"])
    
    print(response)

if __name__ == "__main__":
    main()