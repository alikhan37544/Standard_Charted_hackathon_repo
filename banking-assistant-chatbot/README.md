# Banking Assistant Chatbot

This project is a banking assistant chatbot that utilizes a local language model (LLM) to help users with banking-related tasks. The chatbot can assist users in creating new bank accounts, setting up savings accounts, and opening loans.

## Project Structure

```
banking-assistant-chatbot
├── src
│   ├── main.py          # Entry point of the application
│   ├── utils
│   │   └── helpers.py   # Utility functions for processing user input
│   └── prompts
│       └── templates.py  # Prompt templates for LLM interaction
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd banking-assistant-chatbot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure that the Ollama server is running.

## Usage Guidelines

To run the banking assistant chatbot, execute the following command:

```
python src/main.py
```

Follow the prompts in the console to interact with the chatbot. You can choose from the following options:

1. Create a new bank account
2. Create a new savings account
3. Open a loan

The chatbot will guide you through the process based on your selections.

## Overview of Functionality

The banking assistant chatbot is designed to provide a user-friendly interface for managing banking tasks. It leverages the capabilities of a local LLM to understand user queries and provide relevant responses. The chatbot aims to simplify the banking experience by offering clear guidance and support throughout the process.