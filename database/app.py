import json
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

# Import your DataBaseHandler class
from database_manager import DataBaseHandler

app = Flask(__name__)

CONFIG_FILE = 'database/config.json'
CONFIG_INDEX = 0
UPDATE_INTERVAL = 1800 # 30 minutes

def load_config_from_file(config_file: str) -> dict:
    with open(config_file, 'r') as file:
        return json.load(file)["configs"][CONFIG_INDEX]
    
config = load_config_from_file(CONFIG_FILE)

data_base_handler = DataBaseHandler(
    embeddings_model_config = config['embedding_model_name'],
    embeddings_database_config = config['data_base_handler']
)

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '')
    k = data.get('k', 4)
    docs_and_scores = data_base_handler.search(query, k)
    context = ""
          
    metadata_info = {}
    i=0
    for doc, score in docs_and_scores:
        i += 1
        metadata_info[f"{i}"] = [doc.metadata,
                                    f"accuracy: {(score):.2f}",
                                    doc.page_content]
        
        context += f"{i}.\n{doc.page_content}\n\n"

    return jsonify({'context': context, 'metadata': metadata_info})

def update_database():
    '''
    Function to update the database.
    '''
    data_base_handler.update_database()


#TODO: make the update_database function run every 30 minutes
# # Start the scheduler. This will update the database every 10 seconds
# scheduler = BackgroundScheduler()
# scheduler.add_job(func=update_database, trigger="interval", seconds=UPDATE_INTERVAL)
# scheduler.start()

# # Shut down the scheduler when exiting the app
# atexit.register(lambda: scheduler.shutdown())

if __name__ == '__main__':
    app.run(debug=False, port=5001)
