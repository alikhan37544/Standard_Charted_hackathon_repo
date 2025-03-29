def validate_user_input(user_input):
    valid_options = ["new bank account", "new savings account", "open a loan"]
    return user_input.lower() in valid_options

def format_response(option):
    responses = {
        "new bank account": "You have chosen to create a new bank account.",
        "new savings account": "You have chosen to create a new savings account.",
        "open a loan": "You have chosen to open a loan."
    }
    return responses.get(option, "I'm sorry, I didn't understand that option.")

def extract_option(user_input):
    if "new bank account" in user_input.lower():
        return "new bank account"
    elif "new savings account" in user_input.lower():
        return "new savings account"
    elif "open a loan" in user_input.lower():
        return "open a loan"
    else:
        return None