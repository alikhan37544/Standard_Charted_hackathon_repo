import ollama

class FintechChatbot:
    def __init__(self):
        self.current_domain = None

    def display_fintech_options(self):
        """Display fintech-related options to the user."""
        options = [
            "Want to create a bank account?",
            "Want to get a loan?",
            "Want to delete a bank account?"
        ]
        return options

    def set_domain(self, user_choice):
        """Set the chatbot's domain based on user choice."""
        if "create a bank account" in user_choice.lower():
            self.current_domain = "bank account creation expert"
            return "You're now talking to a bank account creation expert."
        elif "get a loan" in user_choice.lower():
            self.current_domain = "loan expert"
            return "You're now talking to a loan expert."
        elif "delete a bank account" in user_choice.lower():
            self.current_domain = "bank account deletion expert"
            return "You're now talking to a bank account deletion expert."
        else:
            self.current_domain = None
            return "Sorry, I didn't understand your choice."

    def generate_response(self, user_input):
        """Generate a response using the Ollama API and deepseek-r1 model."""
        if self.current_domain:
            # Add domain context to the user input
            context = f"You are a {self.current_domain}. {user_input}"
            response = ollama.generate(
                model="tinyllama",
                prompt=context
            )
            return response["response"]
        else:
            return "Please select a fintech option to proceed."

    def chat(self):
        """Start the chatbot interaction."""
        print("Welcome to the Fintech Chatbot!")
        options = self.display_fintech_options()
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        user_choice = input("Select an option (1/2/3): ")
        if user_choice.isdigit() and 1 <= int(user_choice) <= 3:
            selected_option = options[int(user_choice) - 1]
            print(self.set_domain(selected_option))

            while True:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break
                response = self.generate_response(user_input)
                print(f"Chatbot: {response}")
        else:
            print("Invalid choice. Please try again.")