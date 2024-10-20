from dataclasses import dataclass, asdict
import os
import json
import typing  as t
from moviepy.editor import VideoFileClip
import torch
from transformers import pipeline


CHUNK_LENGTH_SEC = 30

@dataclass
class TranscribedSegment:
    offset_start: str
    offset_end: str
    text: str
    lang: str
    media_type: str
    ref: str
    course_name: str

    def to_dict(self):
        return asdict(self)


class MP4AudioTranscriber:
    '''
    Class to transcribe audio from an mp4 file using the Ivrit-ai whisper model. 
    For more info about Ivrit-ai: https://huggingface.co/ivrit-ai/whisper-v2-d3-e3
    '''

    def __init__(self):
        '''
        Initialize the pipeline for automatic speech recognition using the Ivrit-ai whisper model.

        Attributes:
            device (str): Device to use for the pipeline (cuda:0 if GPU is available, else cpu).
            pipe (AutomaticSpeechRecognitionPipeline): Transformers library pipeline for automatic speech recognition.
        '''
        #TODO: checkout whispher fast model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
                        "automatic-speech-recognition",
                        model="ivrit-ai/whisper-v2-d3-e3",
                        chunk_length_s=CHUNK_LENGTH_SEC,
                        device=self.device,
                        )


    def extract_audio(self, mp4_path: str) -> str:
        ''' Extract audio from the mp4 file into a temporary WAV file, and return the path to the WAV file. '''
        temp_wav_audio_path = mp4_path[:-4] + 'temp_audio.wav'
        video = VideoFileClip(mp4_path)
        video.audio.write_audiofile(temp_wav_audio_path, codec='pcm_s16le')
        return temp_wav_audio_path


    def format_time(self, seconds: float) -> str:
        '''Convert seconds to HH:MM:SS format'''
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:06.3f}"


    def transcribe_audio(self, temp_wav_audio_path: str, url: str, course_name: str) -> t.List[TranscribedSegment]:
        '''
        Transcribe the audio file and return the transcribed segments. 

        Args:
            temp_wav_audio_path (str): Path to the temporary wav audio file.
            url (str): URL of the mp4 file (for reference).
            course_name (str): Name of the course.

        Returns:
            List[TranscribedSegment]: List of transcribed segments.
        '''
        prediction = self.pipe(temp_wav_audio_path, batch_size=8, return_timestamps=True)
        segments = []

        for chunk in prediction.get('chunks', []):
            start_time, end_time = chunk['timestamp']         
            segments.append(TranscribedSegment(
                                            offset_start=self.format_time(start_time),
                                            offset_end=self.format_time(end_time),
                                            text=chunk['text'],
                                            lang='he',
                                            media_type='audio',
                                            ref=url,
                                            course_name=course_name
                                            ))
        return segments


    def save_json(self, data: t.List[TranscribedSegment], output_path: str) -> None:
        '''Save the transcribed segments to a JSON file'''
        formatted_data = [transcrip.to_dict() for transcrip in data]
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump({"data": formatted_data}, json_file, ensure_ascii=False, indent=2)


    def run(self, mp4_path: str, output_path: str, url: str, course_name: str) -> None:
        '''
        Main function to transcribe audio from an mp4 file and save the transcription to a JSON file.
        After transcription, the mp4 and temporary wav audio files are deleted.

        Args:
            mp4_path (str): Path to the mp4 file.
            output_path (str): Path to the output JSON file.
            url (str): URL of the mp4 file (for reference).
            course_name (str): Name of the course.

        Saves:
            JSON file containing the transcribed segments.
        '''
        temp_wav_audio_path = self.extract_audio(mp4_path)
        segments = self.transcribe_audio(temp_wav_audio_path, url, course_name)
        self.save_json(segments, output_path)

        os.remove(mp4_path)
        os.remove(temp_wav_audio_path)
        