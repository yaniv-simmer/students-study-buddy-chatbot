'''
This script requires GPU to run(at a satisfiable speed). I couldn't run it on my local machine so I ran it on Google Colab.
The Colab notebook is available at: https://colab.research.google.com/drive/1mxS2cXstUGyaS3a-VQssKo_y-VbSdaz9?usp=sharing
'''

import json
import os
from time import sleep

from mp4_transcriber import MP4AudioTranscriber

SLEEP_TIME = 60
MP4_UPLOADS_FOLDER = '/shared_folder/mp4_uploads/'
RAW_OUTPUT_FOLDER = '/shared_folder/raw_transcriptions/'
PROCESSED_OUTPUT_FOLDER = '/shared_folder/chunked_transcriptions/'


if __name__ == '__main__':
    mp4_transcriber = MP4AudioTranscriber()
    while True:
        for file in os.listdir(MP4_UPLOADS_FOLDER):        
            segments = mp4_transcriber.run(MP4_UPLOADS_FOLDER + file, RAW_OUTPUT_FOLDER + file[:-4] + '_Ivrit.json')
            os.remove(MP4_UPLOADS_FOLDER + file)
        
        sleep(SLEEP_TIME)
                

