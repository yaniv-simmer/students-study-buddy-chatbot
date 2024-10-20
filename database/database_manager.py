import json
import os
import typing as t

from langchain_community.vectorstores import FAISS, Pinecone
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
import faiss

from chunking_manager import ChunkingManager

PINCONE_ENVIRONMENT = 'us-west1-gcp'
PINCONE_INDEX_NAME = 'langchain-rag'
RAW_TRANSCRIPTION_FOLDER = 'shared_folder/raw_transcriptions/'
DONE_TRANSCRIPTIONS_FILE = 'shared_folder/done_transcriptions.json'
DEFAULT_CHUNK_SIZE = 1000
MAX_K_RESULTS = 3


class VectorStore:
    '''
    A class to represent the VectorStore that stores the documents and their embeddings.
    Can be used with different databases like FAISS, Pinecone, Milvus, Chorma, Elasticsearch.
    '''

    def __init__(self, database_config: str, model_config: str):
        self.database_config = database_config  # TODO: try diffrent databases

        embeddings_model = HuggingFaceEmbeddings(model_name=model_config)
        index = faiss.IndexFlatL2(len(embeddings_model.embed_query(EXAMPLE_QUERY)))
        self.index = FAISS(embedding_function=embeddings_model,
                           index=index,
                           docstore=InMemoryDocstore(),
                           normalize_L2=True,
                           distance_strategy=DistanceStrategy.COSINE,  # TODO: try diffrent distance strategies
                           index_to_docstore_id={}
                           )

    def add_documents(self, documents: t.List[Document]):
        ''' Async run more documents through the embeddings and add to the vectorstore. '''
        self.index.add_documents(documents=documents)

    def similarity_search_with_score(self, query: str, course_name: str, k: int) -> t.List[t.Tuple[Document, float]]:
        '''
        Perform similarity search with a query and return the top k results with their normelized scores.

        Args:
            query (str): The query to search for.
            k (int): The number of results to return.

        Returns:
            list[tuple[Document, float]]: A list of tuples containing the document and the normelized similarity score.
        '''
        docs_and_scores = self.index.similarity_search_with_relevance_scores(query=query,
                                                                             k=k,
                                                                             filter={"course_name": course_name}
                                                                             )
        normelized_docs_and_scores = [(doc, self.normalize_score(score)) for doc, score in docs_and_scores]
        return normelized_docs_and_scores

    def normalize_score(self, cosine_similarity: float) -> float:
        """
        Normalizes cosine similarity from the range [-1, 1] to the range [0, 1].

        Args:
            cosine_similarity (float): The cosine similarity value in the range [-1, 1].

        Returns:
            float: Normalized cosine similarity in the range [0, 1].
        """
        return cosine_similarity

    # return 1 - (cosine_similarity + 1) / 2

    def _initialize_pinecone(self):
        pinecone_api_key = os.environ.get('PINECONE_API_KEY')
        pinecone.init(
            api_key=pinecone_api_key,
            environment=PINCONE_ENVIRONMENT
        )

        if PINCONE_INDEX_NAME not in pinecone.list_indexes():
            dimension = len(self.embeddings.embed_query("hi world"))
            pinecone.create_index(
                name=PINCONE_INDEX_NAME, metric="cosine", dimension=dimension
            )
            index = pinecone.Index(PINCONE_INDEX_NAME)
            Pinecone.from_documents(
                self.documents, self.embeddings, index_name=PINCONE_INDEX_NAME
            )
        return Pinecone.from_existing_index(PINCONE_INDEX_NAME, self.embeddings)


class DBManager:
    '''
    A class to represent the Database Manager that manages the database and the vector store.
    '''

    # TODO: try Milvus, FAISS, Pinecone, Chorma, Elasticsearch

    def __init__(self, database_config: str, model_config: str):
        self.chunking_manager = ChunkingManager(chunk_size=DEFAULT_CHUNK_SIZE)
        self.vector_store = VectorStore(database_config, model_config)

    def similarity_search_with_score(self, query: str, course_name: str, k: int = MAX_K_RESULTS) -> t.List[t.Dict]:
        '''
        Perform similarity search with a query and return the top k results with their scores.

        Args:
            query (str): The query to search for.
            k (int): The number of results to return.

        Returns:
            list[dict]: A list of dictionaries containing the document and the similarity score.
                dict: {
                    "page_content": str,
                    "metadata": dict,
                    "score": float
                }
        '''
        return self.documents_to_json(self.vector_store.similarity_search_with_score(query=query,
                                                                                     course_name=course_name,
                                                                                     k=k
                                                                                     ))

    def update_database(self):
        ''' Check for new documents in the shared json file and add them to the database. '''
        # TODO: dont add transcriptions that are already in the database.
        documents = self.chunking_manager.genarate_chunked_documents_from_shared_folder()
        self.vector_store.add_documents(documents)
        self.save_done_transcriptions()

    def save_done_transcriptions(self):
        ''' Write the done transcriptions to the shared folder. '''
        done_transcriptions = os.listdir(RAW_TRANSCRIPTION_FOLDER)
        with open(DONE_TRANSCRIPTIONS_FILE, 'w', encoding='utf-8') as json_file:
            json.dump({'done_transcriptions': done_transcriptions}, json_file, indent=4)

    def documents_to_json(self, documents: t.List[t.Tuple[Document, float]]) -> t.List[t.Dict]:
        ''' Convert a list of langchain documents to a json object. '''
        return [{
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score)
        }
            for doc, score in documents]

    def save_database(self):
        ''' Save the database to a file. '''
        pass  # TODO: Implement this method
