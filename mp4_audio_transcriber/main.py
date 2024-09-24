import json
import os
from time import sleep

from chunking_manager import ChunkingManager
from mp4_audio_transcriber import MP4AudioTranscriber

SLEEP_TIME = 60
MP4_UPLOADS_FOLDER = '/shared_folder/mp4_uploads/'
RAW_OUTPUT_FOLDER = '/shared_folder/raw_transcriptions/'
PROCESSED_OUTPUT_FOLDER = '/shared_folder/chunked_transcriptions/'


if __name__ == '__main__':
    mp4_transcriber = MP4AudioTranscriber()
    chunking_manager = ChunkingManager()
    while True:
        for file in os.listdir(MP4_UPLOADS_FOLDER):        
            segments = mp4_transcriber.run(MP4_UPLOADS_FOLDER + file, RAW_OUTPUT_FOLDER)
            chunked_segments = chunking_manager.chunk_text(segments)
            os.remove(MP4_UPLOADS_FOLDER + 'mp4_files/' + file)
        
        for file in os.listdir(RAW_OUTPUT_FOLDER):
            if file.endswith('.json') and file not in os.listdir(PROCESSED_OUTPUT_FOLDER):
                with open(RAW_OUTPUT_FOLDER + file, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                chunked_data = chunking_manager.chunk_text(,data)

        sleep(SLEEP_TIME)
                

