import json
from typing import List, Dict

CHUNK_SIZE = 1000


class ChunkingManager:
    '''
    Class to manage the chunking of text segments. 
    This is crucial for the embedding model to work properly.
    '''
    
    def chunk_text(self, output_path: str, segments: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        '''
        Chunk the text segments into CHUNK_SIZE segments.

        Args:
            segments (List[Dict[str, str]]): List of text segments with the following
                keys: 'offset_start', 'offset_end', 'text', 'lang', 'type', 'ref'.
            output_path (str): Path to the output JSON file.

        Returns:
            List[Dict[str, str]]: List of chunked segments with the following keys:
                'offset_start', 'offset_end', 'text', 'lang', 'type', 'ref'.
        '''
        if not segments:
            segments = self.load_json()
        chunked_segments = []
        if not segments:
            return chunked_segments

        current_chunk = {
            'offset_start': None,
            'offset_end': None,
            'text': '',
            'lang': segments[0]['lang'],
            'type': segments[0]['type'],
            'ref': segments[0]['ref']
        }

        for segment in segments:
            segment_text = segment['text']
            if current_chunk['offset_start'] is None:
                current_chunk['offset_start'] = segment['offset_start']

            if len(current_chunk['text']) + len(segment_text) + 1 <= CHUNK_SIZE:
                current_chunk['text'] += (' ' + segment_text) if current_chunk['text'] else segment_text
                current_chunk['offset_end'] = segment['offset_end']
            else:
                chunked_segments.append(current_chunk.copy())
                current_chunk = {
                    'offset_start': segment['offset_start'],
                    'offset_end': segment['offset_end'],
                    'text': segment_text,
                    'lang': segment['lang'],
                    'type': segment['type'],
                    'ref': segment['ref']
                }

        if current_chunk['text']:
            chunked_segments.append(current_chunk)

        self.save_json(chunked_segments, output_path)
        return chunked_segments
    

    def save_json(self, data: List[Dict[str, str]], output_path: str) -> None:
        '''Save the chunked segments to a JSON file'''
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump({"data": data}, json_file, ensure_ascii=False, indent=2)