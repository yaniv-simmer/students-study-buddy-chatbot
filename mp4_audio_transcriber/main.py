'''
This script requires GPU to run(at a satisfiable speed). I couldn't run it on my local machine so I ran it on Google Colab.
The Colab notebook is available at: https://colab.research.google.com/drive/1mxS2cXstUGyaS3a-VQssKo_y-VbSdaz9?usp=sharing
'''

import os
from time import sleep
import typing as t

from mp4_transcriber import MP4AudioTranscriber

SLEEP_TIME = 60
MP4_UPLOADS_FOLDER = '/shared_folder/mp4_uploads/'
RAW_OUTPUT_FOLDER = '/shared_folder/raw_transcriptions/'
PROCESSED_OUTPUT_FOLDER = '/shared_folder/chunked_transcriptions/'

MEDIA_MAPPINGS = {"sample.mp4": {"url": "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file",
                                "course_name": "Sample Course"}
                }
 
def generate_filepath_names(file_name: str) -> t.Tuple[str, str]:
    return (RAW_OUTPUT_FOLDER + file_name[:-4] + '_Ivrit.json', MP4_UPLOADS_FOLDER + file_name)

if __name__ == '__main__':
    mp4AudioTranscriber = MP4AudioTranscriber()
    while True:
        for file in os.listdir(MP4_UPLOADS_FOLDER):
            output_path, mp4_path = generate_filepath_names(file)
            
            # TODO: make this code run in the background for better scaling
            segments = mp4AudioTranscriber.run(
                                            mp4_path=mp4_path,
                                            output_path=output_path,
                                            url=MEDIA_MAPPINGS[file]["url"],
                                            course_name=MEDIA_MAPPINGS[file]["course_name"]
                                            )
        sleep(SLEEP_TIME)
                

