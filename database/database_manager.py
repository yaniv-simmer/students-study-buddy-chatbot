import json
import os
from typing import List, Dict, Any

from langchain.schema import Document
from langchain.vectorstores import FAISS, Pinecone
from langchain.embeddings import HuggingFaceEmbeddings


SHARE_FOLDER_PATH = 'shared_folder/chunked_transcriptions/combined_transcriptions_chunks.json'

class DataLoader:
    ''' Class to load data from a JSON file. '''

    def read_json_file(self) -> List[List[Dict[str, Any]]]:
        ''' Load data from a JSON file. '''
        #TODO: run the chunking script before running this method
        with open(SHARE_FOLDER_PATH, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data['data']


    def data_to_document_objects(self, data: Dict[str, List[Dict[str, str]]]) -> List[Document]:
        '''
        Transforms data into a list of Document objects.
    
        Args:
            data (Dict[str, List[Dict[str, str]]]): Data to be transformed.
    
        Returns:
            List[Document]: List of Document objects.
            Each document object contains the next attributes:
            - page_content: str
            - metadata: Dict[str, Any]
            - metadata contains the next attributes:
                - type: str
                - ref: str
                - offset_start: str
                - offset_end: str
                - lang: str
        '''
        documents = []
        for item in data:
            #TODO: simplify this by just saving the item as is minus the text
            offset_start = item.get('offset_start')
            offset_end = item.get('offset_end')
            text = item.get('text')
            lang = item.get('lang')
            item_type = item.get('type')
            item_ref = item.get('ref')
            metadata = {
                'type': item_type,
                'ref': item_ref,
                'offset_start': offset_start,
                'offset_end': offset_end,
                'lang': lang
            }
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        return documents
    

    def process_data_from_shared_folder(self) -> List[Document]:
        ''' 
        Load data from a JSON file and process it into a list of Document objects.

        Returns:
            List[Document]: List of Document objects.
        '''
        data = self.read_json_file()
        return self.data_to_document_objects(data)




PINCONE_ENVIRONMENT = 'us-west1-gcp'
PINCONE_INDEX_NAME = 'langchain-rag'

class DataBaseHandler:
    def __init__(self, embeddings_model_config: str, embeddings_database_config: str):
        self.data_loader = DataLoader()
        self.embeddings_model = self.initialize_embedding_handler(embeddings_model_config)
        self.embeddings_database = self.initialize_embeddings_database(embeddings_database_config)


    def initialize_embedding_handler(self, embeddings_model_config=None):
        
        # use the default faiss is used as the embeddings model, if no model is specified
        if embeddings_model_config is None:
            return None 
        return HuggingFaceEmbeddings(model_name=embeddings_model_config)
        

    def initialize_embeddings_database(self, embeddings_database_config):
        '''
        Initialize the embeddings database based on the configuration. 

        Returns:
            ???
        '''


        #TODO: check if a SQL is already saved
        documents = self.data_loader.process_data_from_shared_folder()

        if documents == []:
            raise ValueError("No documents to add to the database.")
        
        if embeddings_database_config == 'faiss':
            database = FAISS.from_documents(documents, self.embeddings_model)
        elif embeddings_database_config == 'pinecone':
            database = self.initialize_pinecone()
        else:
            raise ValueError(f"Unsupported vector store type: {self.config}")
        
        self.remove_processed_transcriptions()
        return database


    def initialize_pinecone(self):
        pinecone_api_key = os.environ.get('PINECONE_API_KEY')
        pinecone.init(
            api_key=pinecone_api_key,
            environment=PINCONE_ENVIRONMENT
        )

        if PINCONE_INDEX_NAME not in pinecone.list_indexes():
            dimension = len(self.embeddings.embed_query("Test")) #TODO: What is this test?
            pinecone.create_index(
                name=PINCONE_INDEX_NAME, metric="cosine", dimension=dimension)
            index = pinecone.Index(PINCONE_INDEX_NAME)
            Pinecone.from_documents(
                self.documents, self.embeddings, index_name=PINCONE_INDEX_NAME)
        return Pinecone.from_existing_index(PINCONE_INDEX_NAME, self.embeddings)
            

    def search(self, query: str, k: int = 4):
        return self.embeddings_database.similarity_search_with_score(query, k=k)
    
    
    def remove_processed_transcriptions(self):
        '''
        Remove all chunked transcriptions from the shared file.
        '''
        with open(SHARE_FOLDER_PATH, "r+") as file:
            data = json.load(file)
            data['data'] = []
            file.seek(0)
            json.dump(data, file, indent=2)
            file.truncate()

        
    def update_database(self):
        '''
        Check for new documents in the shared json file and add them to the database.
        '''
        new_documents = self.data_loader.process_data_from_shared_folder()
        if new_documents:
            #TODO: Test this method !!!
            self.embeddings_database.add_documents(new_documents) 
            self.remove_processed_transcriptions()


    def save_database(self):
        '''
        Save the SQL database to a file.
        '''
        pass #TODO: Implement this method


        


