from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from chat_bot import ChatBot

app = Flask(__name__)
CORS(app)

chatbot = ChatBot(enable_gemini=True)


@app.route('/')
def index():
    return send_from_directory('', 'index.html')


@app.route('/api/answer_question', methods=['GET'])
def ask_question():
    """
    Flask API endpoint to get an answer from the chatbot.

    The request should contain a JSON body with a "question" field:
    {
        "question": "What is your question?"
        "course_name": "course_name"
    }

    The response will return the chatbot's answer, metadata, and elapsed time in JSON format:
    {
        "answer",
        "metadata_list":[  
            {
                "metadata": {
                    "course_name", 
                    "lang",
                    "media_type",
                    "offset_end",
                    "offset_start",
                    "ref",
                },
                "page_content",
                "score"
            },
            ...
        ]
    } 
    """
    question = request.args.get('question', '')
    course_name = request.args.get('course_name', '')
    if not question or not course_name:
        return jsonify({"error": "Please provide a question and a course name."}), 400

    answer, metadata_info = chatbot.answer_question(question, course_name)
    return jsonify({
        "answer": answer.content,
        "metadata_list": metadata_info,
    })


if __name__ == '__main__':
    app.run(debug=True)
