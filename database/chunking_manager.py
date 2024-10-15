import json
import os
import typing as t

from langchain.schema import Document


RAW_TRANSCRIPTION_FOLDER = 'shared_folder/raw_transcriptions/'

class DataLoader:
    ''' Class to load data from a JSON file. '''
    
    def load_json(self, file_path: str) -> t.List[t.Dict[str, str]]:
        ''' Load raw transcriptions from the shared folder. '''
        with open(file_path, 'r', encoding='utf-8') as json_file:
            return json.load(json_file)['data']
        
    def load_transcriptions_segments(self, folder_path: str) -> t.List[t.Dict[str, str]]:
        ''' Load all raw transcriptions from the shared folder. '''
        segments = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            segments.extend(self.load_json(file_path))
        return segments


class ChunkingManager:
    ''' Class to chunk text segments '''
    def __init__(self, chunk_size: int = 1000): #TODO: change the default value, why 1000?
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


    def convert_segment_dicts_to_documents(
            self,
            data: t.Dict[str, t.List[t.Dict[str, str]]]
            ) -> t.List[Document]:
        ''' Convert a dictionary of segment dictionaries to a list of Document objects. '''
        documents = []
        for item in data:
            text = item.get('text')
            metadata = item.copy()
            metadata.pop('text')

            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        return documents


    def group_segments_by_course(self, segments: t.List[t.Dict]) -> t.Dict[str, t.List[t.Dict]]:
        '''
        Group the segments by course name.
        
        Args:
            segments (List[Dict]): List of text segments with the following
                keys: 'offset_start','offset_end', 'text', 'lang', 'type', 'ref', 'course_name'.

        Returns:
            Dict[str,List[Dict]]: Dictionary with course names as keys and lists of segments as values.
        '''
        segments_by_course = {}
        for segment in segments:
            course_name = segment['course_name']
            if course_name not in segments_by_course:
                segments_by_course[course_name] = []
            segments_by_course[course_name].append(segment)
        return segments_by_course 


    def genarate_chunked_documents_from_shared_folder(self) -> t.Dict[str, t.List[Document]]:
        '''
        Generate chunked documents from the transcriptions in the shared folder. 

        Returns:
            Dict[str,List[Document]]: Dictionary with course names as keys and lists of Document objects as values.
                Each document object contains the next attributes:
                    - page_content: str
                    - metadata: Dict[str,Any]
                        - "offset_start"
                        - "offset_end"
                        - "lang"
                        - "ref"
                        - "course_name"
                        - "media_type"  
        '''
        raw_segments = self.data_loader.load_transcriptions_segments(RAW_TRANSCRIPTION_FOLDER)
        segments_by_course = self.group_segments_by_course(raw_segments)
        chunked_documents = {
            course_name : self.convert_segment_dicts_to_documents(
                self.split_text_into_chunked_segments(segments)
            )
            for course_name, segments in segments_by_course.items()
        }

        return chunked_documents