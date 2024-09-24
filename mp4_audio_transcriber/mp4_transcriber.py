import os
import json
import argparse
from typing import List, Dict
from moviepy.editor import VideoFileClip
import torch
from transformers import pipeline


CHUNK_LENGTH_SEC = 30

class MP4AudioTranscriber:
    '''
    Class to transcribe audio from an mp4 file using the Ivrit-ai whisper model. 
    For more info: https://huggingface.co/ivrit-ai/whisper-v2-d3-e3
    '''
    def __init__(self):
        self.mp4_path = None
        self.output_path = None
        self.temp_audio_path = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="ivrit-ai/whisper-v2-d3-e3",
            chunk_length_s=CHUNK_LENGTH_SEC,
            device=self.device,
        )

    def extract_audio(self) -> None:
        '''Extract audio from the mp4 file'''
        video = VideoFileClip(self.mp4_path)
        video.audio.write_audiofile(self.temp_audio_path, codec='pcm_s16le')

    def format_time(self, seconds: float) -> str:
        '''Convert seconds to HH:MM:SS format'''
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:06.3f}"

    def transcribe_audio(self) -> List[Dict[str, str]]:
        '''
        Transcribe the audio file and return the transcribed segments. 

        Returns:
            List[Dict[str, str]]: List of transcribed segments with the following keys:
                'offset_start', 'offset_end', 'text', 'lang', 'type', 'ref'.
        '''
        prediction = self.pipe(self.temp_audio_path, batch_size=8, return_timestamps=True)
        segments = []

        for chunk in prediction.get('chunks', []):
            start_time, end_time = chunk['timestamp']
            segments.append({
                "offset_start": self.format_time(start_time),
                "offset_end": self.format_time(end_time),
                "text": chunk['text'].strip(),
                "lang": "he", #TODO: add language detection
                "type": "audio",
                "ref": self.mp4_path
            })

        return segments


    def save_json(self, data: List[Dict[str, str]]) -> None:
        '''Save the transcribed segments to a JSON file'''
        with open(self.output_path, 'w', encoding='utf-8') as json_file:
            json.dump({"data": data}, json_file, ensure_ascii=False, indent=2)

    def run(self, mp4_path: str, output_folder: str) -> List[Dict[str, str]]:
        '''
        Main function to transcribe audio from an mp4 file and save the transcription to a JSON file.

        Args:
            mp4_path (str): Path to the mp4 file.
            output_folder (str): Path to the output folder where the JSON file will be saved.

        Returns:
            List[Dict[str, str]]: List of transcribed segments with the following keys:
                'offset_start', 'offset_end', 'text', 'lang', 'type', 'ref'.
        '''
        self.mp4_path =  mp4_path
        self.output_path = output_folder + mp4_path[:-4] + '_Ivrit.json'
        self.temp_audio_path = mp4_path[:-4] + 'temp_audio.wav'

        print("Extracting audio...")
        self.extract_audio()

        print("Transcribing audio...")
        segments = self.transcribe_audio()

        print("Saving transcription to JSON...")
        self.save_json(segments)

        os.remove(self.temp_audio_path)
        print("Done.")

        return segments


