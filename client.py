import tkinter as tk
from tkinter import scrolledtext
import requests
import json

class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot Client")

        # Create a label and input box for the user question
        self.label = tk.Label(root, text="Ask the chatbot a question:")
        self.label.pack(pady=10)

        self.question_input = tk.Entry(root, width=50)
        self.question_input.pack(pady=10)

        # Button to submit the question
        self.ask_button = tk.Button(root, text="Ask", command=self.get_answer)
        self.ask_button.pack(pady=10)

        # Create a scrolled text area to display the chatbot's response
        self.response_area = scrolledtext.ScrolledText(root, width=60, height=20, wrap=tk.WORD)
        self.response_area.pack(pady=10)

    def get_answer(self):
        
        # Get the question from the input box
        question = self.question_input.get()

        # URL of the Flask API endpoint
        api_url = "http://127.0.0.1:5000/api/answer"

        # Create a payload with the question
        payload = {"question": question}

        try:
            # Send the POST request to the Flask API
            response = requests.post(api_url, json=payload)
            response_data = response.json()

            # Extract the answer, metadata, and elapsed time from the response
            answer = response_data.get("answer", "No answer found")
            metadata = response_data.get("metadata", {})
            elapsed_time = response_data.get("elapsed_time", 0)

            # Clear the response area before displaying new content
            self.response_area.delete(1.0, tk.END)

            # Display the chatbot's answer and elapsed time
            self.response_area.insert(tk.INSERT, f"Answer: {answer}\n\n")
            self.response_area.insert(tk.INSERT, f"Time taken: {elapsed_time:.2f} seconds\n\n")

            # Display metadata info if available
            if metadata:
                self.response_area.insert(tk.INSERT, "Metadata:\n")
                for key, value in metadata.items():
                    self.response_area.insert(tk.INSERT, f"Document {key}:\n")
                    self.response_area.insert(tk.INSERT, f" - Type: {value[0]['type']}\n")
                    self.response_area.insert(tk.INSERT, f" - Reference: {value[0]['ref']}\n")
                    self.response_area.insert(tk.INSERT, f" - : {value[1]}\n")
                    self.response_area.insert(tk.INSERT, f" - Content: {value[2]}\n\n")
            else:
                self.response_area.insert(tk.INSERT, "No metadata available.\n")

        except requests.exceptions.RequestException as e:
            self.response_area.insert(tk.INSERT, f"Error: {e}")

if __name__ == "__main__":
    # Create the main Tkinter window
    root = tk.Tk()
    
    # Create the ChatbotApp instance
    app = ChatbotApp(root)

    # Run the Tkinter event loop
    root.mainloop()
