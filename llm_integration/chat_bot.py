import json
import time
from langchain.llms import HuggingFaceHub
from typing import List, Dict, Any, Tuple
from langchain import PromptTemplate
from dotenv import load_dotenv
import os

import requests
load_dotenv()

URL = 'http://localhost:5001/search'




class LLMHandler:
    def __init__(self, llm_model_name: str, huggingface_model_kwargs: Dict[str, Any] = {}):
        self.llm = self.initialize_llm(llm_model_name, huggingface_model_kwargs)

    def initialize_llm(self, llm_model_name, huggingface_model_kwargs):
        
        huggingface_api_key = os.environ.get('HUGGINGFACE_API_KEY')
        
        return HuggingFaceHub(
            repo_id=llm_model_name,
            model_kwargs=huggingface_model_kwargs,
            huggingfacehub_api_token=huggingface_api_key
        )

    def generate_answer(self, prompt_text: str) -> str:
        return self.llm(prompt_text)
    


# **PromptManager** class
class PromptManager:
    def __init__(self, language_config: str):
        self.template = self.create_prompt_template(language_config)

    def create_prompt_template(self, language_config) -> PromptTemplate:
        if language_config == 'english':
            template = config.get('prompt_template', """
You are a lecturer. The user will ask you questions. Use the following context to answer the question.
If you don't know the answer, just say you don't know.
No longer than 2 sentences.

Context: {context}

Question: {question}

""")
        else:
            template = config.get('prompt_template', """
אתה מרצה באוניברסיטה. המשתמש ישאל אותך שאלות. השתמש רק באחד מההקשרים הבאים כדי לענות על השאלה.
אם אינך יודע את התשובה, פשוט אמור שאינך יודע.
ספק תשובה קצרה ותמציתית, לא יותר מ-2 משפטים.


הקשר:
{context}

השאלה של המשתמש: 
{question}

""")
        
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def format_prompt(self, context: str, question: str) -> str:
        return self.template.format(context=context, question=question)



class ChatBot:
    '''
    A class to represent the ChatBot.

    '''
    def __init__(self, llm_model_name: str, language_config: str, model_kwargs: Dict[str, Any] = {}):
        self.llm_handler = LLMHandler(llm_model_name, model_kwargs)
        self.prompt_manager = PromptManager(language_config)
    
    def answer_question(self, question: str) -> Tuple[str, Dict[str, List[str]], float]:
        '''
        Answer a user question and return the answer, metadata info, and elapsed time.

        Args:
            question (str): The user's question.

        Returns:
            str: The LLM answer to the user's question.
            Dict[str, List[Union[Dict[str, Any], str]]]: Metadata information of the 3 sources with the highest similarity scores.
                - The dictionary keys represent the rank of each document based on the similarity score (e.g., "1", "2", "3").
                - Each key maps to a list of three elements:
                    1. A dictionary (`Dict[str, Any]`) containing the metadata for the document. This typically includes:
                        - 'type': The type of the document (e.g., "pdf", "text", "web").
                        - 'ref': A reference to the document, such as a URL, document ID, or file path.
                        - Any other metadata fields associated with the document, depending on the source.
                    2. A string representing the accuracy or similarity score, formatted as "accuracy: <normalized_score>".
                    3. The page content (string) extracted from the document, processed and formatted for display (e.g., a summary or excerpt).

            float: The elapsed time for the process, in seconds.
        '''
        start_time = time.time()

        data = {'query': question, 'k': 3}
        
        response = requests.post(URL, json=data)
        if response.status_code != 200:
            return f"An error occurred: {response.text}", {}, 0.0
        response_data = response.json()
        context = response_data['context']
        metadata_info = response_data['metadata']        
        

        prompt_text = self.prompt_manager.format_prompt(context=context, question=question)

        answer = self.llm_handler.generate_answer(prompt_text)

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        return answer, metadata_info, elapsed_time
      

    def normalize_cosine_similarity(self, cosine_similarity: float) -> float:
        return (cosine_similarity + 1) / 2


from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize the ChatBot instance

CONFIG_FILE = 'llm_integration/config.json'
CONFIG_INDEX = 0

def load_config_from_file(config_file: str) -> dict:
    with open(config_file, 'r') as file:
        return json.load(file)["configs"][CONFIG_INDEX]
    
config = load_config_from_file(CONFIG_FILE)

chatbot = ChatBot(llm_model_name=config['llm_model_name'], language_config=config['language'],
                   model_kwargs=config['huggingface_model_kwargs'])

@app.route('/api/answer', methods=['POST'])
def answer_question():
    """
    Flask API endpoint to get an answer from the chatbot.

    The request should contain a JSON body with a "question" field:
    {
        "question": "What is your question?"
    }

    The response will return the chatbot's answer, metadata, and elapsed time in JSON format:
    {
        "answer": "Chatbot's answer",
        "metadata": { ... },  # Metadata information of the top documents
        "elapsed_time": 1.23   # Time taken to process the request
    }
    """
    # Extract the user question from the request
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "Please provide a question."}), 400

    # Get the answer, metadata info, and elapsed time from the chatbot
    answer, metadata_info, elapsed_time = chatbot.answer_question(question)

    # Return the result as a JSON response
    return jsonify({
        "answer": answer,
        "metadata": metadata_info,
        "elapsed_time": elapsed_time
    })


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
