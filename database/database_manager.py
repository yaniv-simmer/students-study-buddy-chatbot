import json
from typing import List, Dict, Any

from langchain.schema import Document
from langchain.vectorstores import FAISS, Pinecone
from langchain.embeddings import HuggingFaceEmbeddings


class DataLoader:
    ''' Class to load data from a JSON file. '''
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> List[List[Dict[str, Any]]]:
        ''' Load data from a JSON file. '''
        with open(self.file_path, "r", encoding='utf-8') as file:
            data = json.load(file)
        return data

    def data_to_documents(self, data: Dict[str, List[Dict[str, str]]]) -> List[Document]:
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


# **EmbeddingHandler** class
class EmbeddingHandler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = self.initialize_embeddings()

    def initialize_embeddings(self):
        embedding_type = self.config['embedding_type']
        if embedding_type == 'huggingface':
            model_name = self.config.get('embedding_model_name')
            return HuggingFaceEmbeddings(model_name=model_name)
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")


# TODO : save the vector store to a file and load it from the file
class VectorStoreHandler:
    def __init__(self, config: Dict[str, Any], embeddings, documents: List[Document]):
        self.config = config
        self.embeddings = embeddings
        self.documents = documents
        self.vector_store = self.initialize_vector_store()

    def initialize_vector_store(self):
        vector_store_type = self.config['vector_store_type']
        if vector_store_type == 'faiss':
            return FAISS.from_documents(self.documents, self.embeddings)
        elif vector_store_type == 'pinecone':
            return self.initialize_pinecone()
        else:
            raise ValueError(f"Unsupported vector store type: {vector_store_type}")

    def initialize_pinecone(self):
        pinecone_api_key = os.environ.get('PINECONE_API_KEY')
        pinecone.init(
            api_key=pinecone_api_key,
            environment=self.config.get('pinecone_environment', 'us-west1-gcp')
        )
        index_name = self.config.get('index_name', 'langchain-rag')

        if index_name not in pinecone.list_indexes():
            dimension = len(self.embeddings.embed_query("Test"))
            pinecone.create_index(
                name=index_name, metric="cosine", dimension=dimension)
            index = pinecone.Index(index_name)
            Pinecone.from_documents(
                self.documents, self.embeddings, index_name=index_name)
        return Pinecone.from_existing_index(index_name, self.embeddings)

    def similarity_search_with_score(self, query: str, k: int = 4):
        return self.vector_store.similarity_search_with_score(query, k=k)
