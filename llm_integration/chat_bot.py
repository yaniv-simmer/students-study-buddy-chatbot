import requests
import json
import os
import typing as t
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate

load_dotenv()

CONFIG_INDEX = 1
DB_SEARCH_ENDPOINT = 'http://localhost:5001/similarity_search_with_score'
PROMPT_TEMPLATE = (
    "You are a lecturer. The user will ask you questions. Use the following list of contexts to answer the question."
    "answer the question in Hebrew. No longer than 2 sentences."
    "Also refer to the context that was most relavent to the question. the last character should be the contex number that was most relavent"
    "\nContext: {context}"
    "\nQuestion: {question}"
)


class ChatBot:
    '''
    A class to represent the ChatBot.
    '''

    def __init__(self, enable_gemini: bool):
        self.llm = self.initialize_llm(enable_gemini)
        self.prompt_template = PromptTemplate(template=PROMPT_TEMPLATE,
                                              input_variables=["context", "question"]
                                              )

    def initialize_llm(self, enable_gemini: bool) -> t.Union[HuggingFaceHub, ChatGoogleGenerativeAI]:
        ''' Initialize the LLM model based on the configuration file or the gemini-1.5-flash model. '''
        if enable_gemini:
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                          temperature=0.1,
                                          max_tokens=None,
                                          timeout=None,
                                          max_retries=2,
                                          )
        else:
            configs_lst = self.load_config_from_file()
            config = configs_lst[CONFIG_INDEX]
            return HuggingFaceHub(repo_id=config["llm_model_name"],
                                  model_kwargs=config["huggingface_model_kwargs"],
                                  huggingfacehub_api_token=os.environ.get('HUGGINGFACE_API_KEY')
                                  )

    def answer_question(self, query: str, course_name: str) -> t.Tuple[str, t.List[t.Dict]]:
        '''
        Answer a question based on the user query and the course name.

        Args:
            query (str): The user query.
            course_name (str): The course name.
            k (int): The number of results to return.

        Returns:
            tuple[str, list[dict]]: The answer and the retrieved data
                str: The answer to the question.
                list[dict]: The retrieved data from the database.
                                dict: {
                                    "page_content": str,
                                    "metadata": dict,
                                    "score": float
                                }
        '''
        response = requests.get(DB_SEARCH_ENDPOINT, params={'query': query, "course_name": course_name})
        if response.status_code != 200:
            return f"An error occurred: {response.text}", {}
        response_data = response.json()

        db_data = response_data['docs_and_scores']
        formatted_prompt = self.format_prompt(db_data, query)
        answer = self.llm.invoke(formatted_prompt)
        return answer, db_data

    def format_prompt(self, db_data: str, question: str) -> str:
        ''' Format the prompt based on the retrieved data and the question. '''
        retrived_context = [f"{i + 1}.    {doc['page_content']}" for i, doc in enumerate(db_data)]
        context = '\n\n'.join(retrived_context)
        return self.prompt_template.format(context=context, question=question)

    def load_config_from_file(self, config_file: str) -> t.Dict[str, str]:
        ''' Load the configuration from the configuration file and return a list of optional configs'''
        with open(config_file, 'r') as file:
            return json.load(file)["configs"]
