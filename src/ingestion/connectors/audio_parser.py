import os
import whisper
import speech_recognition as sr
from pydub import AudioSegment

class AudioParser:
    def __init__(self, audio_path, model_name="base"):
        self.audio_path = audio_path
        self.model_name = model_name

    def extract_text_with_whisper(self, output_dir="output/audio/"):
        """
        Extract text from audio using OpenAI Whisper.
        :param output_dir: Directory to save the transcript.
        :return: Path to the saved transcript file.
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load Whisper model
        model = whisper.load_model(self.model_name)
        
        # Perform transcription
        result = model.transcribe(self.audio_path)
        
        # Save transcript
        transcript_path = os.path.join(output_dir, "transcript_whisper.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        return transcript_path

    def extract_text_with_speech_recognition(self):
        """
        Extract text from audio using SpeechRecognition library.
        :return: Transcribed text as a string.
        """
        recognizer = sr.Recognizer()
        
        try:
            with sr.AudioFile(self.audio_path) as source:
                # Record audio data
                audio_data = recognizer.record(source)
            
            # Recognize speech
            text = recognizer.recognize_google(audio_data)
            return text
        
        except sr.UnknownValueError:
            return "Audio is unclear, could not recognize."
        
        except sr.RequestError as e:
            return f"Speech recognition service error: {e}"

    @staticmethod
    def convert_to_wav(input_path, output_path):
        """
        Convert audio file to WAV format with standard settings.
        :param input_path: Path to the input audio file.
        :param output_path: Path to save the converted WAV file.
        :return: None
        """
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)  # Standardize frame rate and channels
        audio.export(output_path, format="wav")
    