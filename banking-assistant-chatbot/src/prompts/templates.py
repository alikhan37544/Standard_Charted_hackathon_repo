from langchain.prompts import PromptTemplate

# Define prompt templates for the banking assistant chatbot
create_account_prompt = PromptTemplate(
    input_variables=["user_name"],
    template="Hello {user_name}, would you like to create a new bank account? Please provide your details."
)

create_savings_account_prompt = PromptTemplate(
    input_variables=["user_name"],
    template="Hi {user_name}, are you interested in creating a new savings account? Please share your information."
)

open_loan_prompt = PromptTemplate(
    input_variables=["user_name"],
    template="Greetings {user_name}, would you like to open a loan? Please tell me more about what you need."
)

# List of available prompts
prompts = {
    "create_account": create_account_prompt,
    "create_savings_account": create_savings_account_prompt,
    "open_loan": open_loan_prompt,
}