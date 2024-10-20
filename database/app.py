from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import json
import typing as t

from langchain.schema import Document

from database_manager import DBManager

app = Flask(__name__)

MAX_K_RESULTS = 5
CONFIG_FILE_PATH = 'database/config.json'
CONFIG_INDEX = 0
UPDATE_INTERVAL = 1800  # 30 minutes


def load_config_from_file(config_file: str) -> dict:
    with open(config_file, 'r') as file:
        return json.load(file)["configs"][CONFIG_INDEX]


config = load_config_from_file(CONFIG_FILE_PATH)
db_manager = DBManager(
    model_config=config['embedding_model_name'],
    database_config=config['db_handler']
)
db_manager.update_database()


@app.route('/similarity_search_with_score', methods=['GET'])
def similarity_search_with_score() -> t.Dict[str, t.List[t.Dict]]:
    '''
    Perform similarity search with a query and return the top k results with their scores.

    The request should contain the following query parameters:
    {
        "query": "What is your question?",
        "course_name": "course_name"
    }

    The response will return the top k results with their scores in JSON format:
    {
        "docs_and_scores": [
            {
                "page_content",
                "metadata": {
                    "offset_start",
                    "offset_end",
                    "lang",
                    "ref",
                    "course_name",
                    "media_type"
                },
                "score"
            },
            ...
        ]
    }
    '''
    query = request.args.get('query', '')
    course_name = request.args.get('course_name', '')

    docs_and_scores = db_manager.similarity_search_with_score(query, course_name)
    return jsonify({'docs_and_scores': docs_and_scores})


def update_database():
    ''' Function to update the database. This function is called by the scheduler at regular intervals. '''
    db_manager.update_database()


scheduler = BackgroundScheduler()
scheduler.add_job(func=update_database, trigger="interval", seconds=UPDATE_INTERVAL)
scheduler.start()

atexit.register(lambda: scheduler.shutdown())

if __name__ == '__main__':
    app.run(debug=False, port=5001)
