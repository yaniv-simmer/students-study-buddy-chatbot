import json
import os
import typing as t

from langchain.schema import Document

RAW_TRANSCRIPTION_FOLDER = 'shared_folder/raw_transcriptions/'
DONE_TRANSCRIPTIONS_FILE = 'shared_folder/done_transcriptions.json'


class DataLoader:
    ''' Class to load data from a JSON file. '''

    def load_json(self, file_path: str) -> t.List[t.Dict[str, str]]:
        ''' Load raw transcriptions from the shared folder. '''
        with open(file_path, 'r', encoding='utf-8') as json_file:
            return json.load(json_file)['data']

    def read_done_transcriptions(self) -> t.List[str]:
        ''' Read the done transcriptions from the shared folder. '''
        # TODO: implemnt this !
        return []

    def load_transcriptions_segments(self, folder_path: str) -> t.List[t.Dict[str, str]]:
        ''' Load only the transcriptions that have not been processed yet. '''
        done_transcriptions_files = self.read_done_transcriptions()
        all_files = os.listdir(folder_path)
        pending_files = set(all_files) - set(done_transcriptions_files)

        segments = []
        for file in pending_files:
            file_path = os.path.join(folder_path, file)
            segments.extend(self.load_json(file_path))
        return segments


class ChunkingManager:
    ''' Class to chunk text segments '''

    def __init__(self, chunk_size: int = 1000):  # TODO: change the default value, why 1000?
        self.chunk_size = chunk_size
        self.data_loader = DataLoader()

    def split_text_into_chunked_segments(
            self,
            segments: t.List[t.Dict] = None
    ) -> t.List[t.Dict]:
        '''
        Chunk the text segments into chunk_size segments.

        Args:
            segments (List[Dict[str, str]]): List of text segments with the followingkeys:
                'offset_start', 'offset_end', 'text', 'lang', 'media_type', 'ref', 'course_name'.

        Returns:
            List[Dict]: List of chunked segments with the same keys as the input segments.
        '''
        # TODO: research if chunking appoximation is good enough, or is there a better way to chunk the text
        chunked_segments = []
        if not segments:
            return chunked_segments

        current_chunk = {
            'offset_start': None,
            'offset_end': None,
            'text': '',
            'lang': segments[0]['lang'],
            'media_type': segments[0]['media_type'],
            'ref': segments[0]['ref'],
            'course_name': segments[0]['course_name']
        }

        for segment in segments:
            segment_text = segment['text']
            if current_chunk['offset_start'] is None:
                current_chunk['offset_start'] = segment['offset_start']

            if len(current_chunk['text']) + len(segment_text) + 1 <= self.chunk_size:
                current_chunk['text'] += (' ' + segment_text) if current_chunk['text'] else segment_text
                current_chunk['offset_end'] = segment['offset_end']
            else:
                chunked_segments.append(current_chunk.copy())
                current_chunk = {
                    'offset_start': segment['offset_start'],
                    'offset_end': segment['offset_end'],
                    'text': segment_text,
                    'lang': segment['lang'],
                    'media_type': segment['media_type'],
                    'ref': segment['ref'],
                    'course_name': segment['course_name']
                }

        if current_chunk['text']:
            chunked_segments.append(current_chunk)

        return chunked_segments

    def create_document_objects(self, data: t.List[t.Dict]) -> t.List[Document]:
        ''' Convert the chunked segments into Document objects. '''
        return [Document(page_content=segment.pop('text'),
                         metadata=segment
                         )
                for segment in data]

    def genarate_chunked_documents_from_shared_folder(self) -> t.List[Document]:
        '''
        Generate chunked documents from the transcriptions in the shared folder. 

        Returns:
            List[Document]]: List of Document objects.
                Each document object contains the next attributes:
                    - page_content: str
                    - metadata: Dict[str,Any]
                        {
                            "offset_start",
                            "offset_end",
                            "lang",
                            "ref",
                            "course_name",
                            "media_type"
                        }
        '''
        # TODO: use the langchain_core.runnablesRunnablePassthrough | operator to simplify the code
        raw_segments = self.data_loader.load_transcriptions_segments(RAW_TRANSCRIPTION_FOLDER)
        chunked_segments = self.split_text_into_chunked_segments(raw_segments)
        chunked_documents = self.create_document_objects(chunked_segments)

        return chunked_documents
