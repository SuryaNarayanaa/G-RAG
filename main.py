from src.ingestion.connectors.pdf_parser import PDFParser
from src.ingestion.connectors.image_parser import ImageParser
from src.ingestion.connectors.audio_parser import AudioParser
from src.ingestion.connectors.video_parser import VideoParser


# First handle PDF parsing
pdf_path = "testing/tables2.pdf"
output_dir = "output/"

parser = PDFParser(pdf_path, output_dir)

table_paths = parser.extract_tables()
print("Extracted tables saved to:", table_paths)

text_file, image_files = parser.extract_text_and_images()
print("Extracted text saved to:", text_file)
print("Extracted images saved to:", image_files)

markdown_path = parser.convert_to_markdown()
print("Markdown file saved to:", markdown_path)

# Now handle any images that were extracted
# for image_file in image_files[0]:
image_parser = ImageParser('D:\\Learning\\projects\\G-RAG\\testing\\image.png', 'D:\\Learning\\projects\\G-RAG\\testing')

# Preprocess the image
preprocessed_image = image_parser.preprocess_image()
if preprocessed_image:
    print(f"Preprocessed image saved to: {preprocessed_image}")

# Extract content from image
image_content = image_parser.extract_text_and_metadata()
if image_content:
    print("Extracted content from {}")

audio_path = "testing/audio.wav"
parser = AudioParser(audio_path)

# Transcribe using Whisper
transcript_path = parser.extract_text_with_whisper()
print(f"Transcript saved to: {transcript_path}")


youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
video_path = VideoParser.download_youtube_video(youtube_url)
print(f"YouTube video downloaded to: {video_path}")


parser = VideoParser(video_path)
transcript_path = parser.transcribe_video(model="whisper")
print(f"Transcript saved to: {transcript_path}")