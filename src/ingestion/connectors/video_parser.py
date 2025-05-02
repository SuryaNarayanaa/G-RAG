import os
from moviepy.editor import VideoFileClip
import yt_dlp
from .audio_parser import AudioParser


class VideoParser:
    def __init__(self, video_path=None):
        self.video_path = video_path

    @staticmethod
    def extract_audio_from_video(video_path, output_dir="output/video/"):
        """
        Extract audio from a video file and save it as a WAV file.
        :param video_path: Path to the video file.
        :param output_dir: Directory to save the extracted audio.
        :return: Path to the extracted WAV file.
        """
        os.makedirs(output_dir, exist_ok=True)
        video = VideoFileClip(video_path)
        audio_path = os.path.join(output_dir, "extracted_audio.wav")
        video.audio.write_audiofile(audio_path, codec="pcm_s16le")
        return audio_path

    @staticmethod
    def download_youtube_video(youtube_url, output_dir="data/youtube/"):
        """
        Download a YouTube video and save it locally using yt-dlp.
        :param youtube_url: URL of the YouTube video.
        :param output_dir: Directory to save the downloaded video.
        :return: Path to the downloaded video file.
        """
        os.makedirs(output_dir, exist_ok=True)
        ydl_opts = {
            "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
            "format": "mp4",
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                return os.path.join(output_dir, f"{info['title']}.mp4")
        except Exception as e:
            print(f"Error downloading YouTube video with yt-dlp: {e}")
            return None
    def transcribe_video(self, output_dir="output/video/", model="whisper"):
        """
        Transcribe audio extracted from a video file.
        :param output_dir: Directory to save transcript.
        :param model: Transcription model to use ('whisper' or 'speech_recognition').
        :return: Path to the transcript file or transcribed text.
        """
        # Extract audio
        audio_path = self.extract_audio_from_video(self.video_path, output_dir)

        # Use audio parser for transcription
        parser = AudioParser(audio_path)

        if model == "whisper":
            return parser.extract_text_with_whisper(output_dir)
        elif model == "speech_recognition":
            return parser.extract_text_with_speech_recognition()
        else:
            raise ValueError("Invalid model specified. Choose 'whisper' or 'speech_recognition'.")
