#!/usr/bin/env python3
"""
RAG-CLI: A user-friendly command line interface for G-RAG
This tool helps you process various file types and perform RAG operations.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.connectors.pdf_parser import PDFParser
from src.ingestion.connectors.audio_parser import AudioParser
from src.ingestion.connectors.image_parser import ImageParser
from src.ingestion.connectors.video_parser import VideoParser
from dotenv import load_dotenv

# Import FAISS and other components for vector search
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

# Define color codes for better terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.GREEN}{text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}{text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}{text}{Colors.ENDC}")

def check_api_key():
    """Check if the Google API key is set."""
    if os.getenv("GOOGLE_API_KEY") is None:
        print_error("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please get an API key from Google AI Studio (https://aistudio.google.com/app/apikey)")
        print("and set it in your .env file or environment variables.")
        return False
    return True

def process_pdf(args):
    """Process a PDF file and extract text, images, and tables."""
    print_header(f"Processing PDF: {args.input}")
    
    output_dir = args.output if args.output else "output/"
    pdf_parser = PDFParser(args.input, output_dir=output_dir)
    
    print("Extracting text and images...")
    text_path, image_paths = pdf_parser.extract_text_and_images()
    print_success(f"Text extracted to: {text_path}")
    print(f"Extracted {len(image_paths)} images")
    
    if args.tables:
        print("Extracting tables...")
        table_paths = pdf_parser.extract_tables()
        print_success(f"Extracted {len(table_paths)} tables")
    
    if args.markdown:
        print("Converting to markdown...")
        markdown_path = pdf_parser.convert_to_markdown()
        print_success(f"Markdown saved to: {markdown_path}")
    
    return {
        "text": text_path,
        "images": image_paths
    }

def process_audio(args):
    """Process an audio file and extract text."""
    print_header(f"Processing Audio: {args.input}")
    
    output_dir = args.output if args.output else "output/audio/"
    audio_parser = AudioParser(args.input)
    
    print("Converting to WAV format...")
    wav_path = os.path.join(output_dir, f"{os.path.basename(args.input)}.wav")
    AudioParser.convert_to_wav(args.input, wav_path)
    
    print("Transcribing audio with Whisper...")
    transcript_path = audio_parser.extract_text_with_whisper(output_dir=output_dir)
    print_success(f"Transcript saved to: {transcript_path}")
    
    return {
        "transcript": transcript_path
    }

def process_image(args):
    """Process an image file and extract text and metadata."""
    print_header(f"Processing Image: {args.input}")
    
    image_parser = ImageParser(args.input)
    print("Extracting text and metadata...")
    
    result = image_parser.extract_text_and_metadata()
    
    # Save the results to a JSON file in the output directory
    output_dir = args.output if args.output else "output/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "image_content.json")
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print_success(f"Extracted text: {result['text'][:100]}...")
    print_success(f"Results saved to: {output_path}")
    
    return result

def process_video(args):
    """Process a video file or YouTube URL and extract audio and transcription."""
    print_header(f"Processing Video: {args.input}")
    
    # Check if the input is a YouTube URL or local file
    if args.input.startswith("http"):
        print("Downloading YouTube video...")
        video_path = VideoParser.download_youtube_video(args.input)
        print_success(f"Video downloaded to: {video_path}")
    else:
        video_path = args.input
    
    output_dir = args.output if args.output else "output/video/"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Extracting audio...")
    audio_path = VideoParser.extract_audio_from_video(video_path, os.path.join(output_dir, "extracted_audio.wav"))
    print_success(f"Audio extracted to: {audio_path}")
    
    print("Transcribing audio with Whisper...")
    transcript_path = AudioParser(audio_path).extract_text_with_whisper(output_dir=output_dir)
    print_success(f"Transcript saved to: {transcript_path}")
    
    return {
        "video": video_path,
        "audio": audio_path,
        "transcript": transcript_path
    }

def create_rag(args):
    """Create a RAG model from the specified documents."""
    print_header("Creating RAG System")
    
    if not check_api_key():
        return
    
    # Initialize the embedding model
    print("Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    print_success("Embedding model initialized.")
    
    # Prepare documents
    print("Preparing documents...")
    documents = []
    
    for doc_path in args.documents:
        try:
            # Check file type and handle accordingly
            if doc_path.endswith('.txt'):
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append(Document(page_content=content, metadata={"source": doc_path}))
                print(f"Added text document: {doc_path}")
            
            elif doc_path.endswith('.json'):
                with open(doc_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and "content" in data:
                    documents.append(Document(page_content=data["content"], 
                                            metadata=data.get("metadata", {"source": doc_path})))
                    print(f"Added JSON document: {doc_path}")
                else:
                    print_warning(f"JSON structure not recognized in {doc_path}, skipping")
            
            elif doc_path.endswith('.pdf'):
                # Extract text using PDF parser
                pdf_parser = PDFParser(doc_path, output_dir="output/")
                text_path, _ = pdf_parser.extract_text_and_images()
                with open(text_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append(Document(page_content=content, metadata={"source": doc_path, "type": "pdf"}))
                print(f"Added PDF document: {doc_path}")
                
            else:
                print_warning(f"Unsupported file type: {doc_path}, skipping")
                
        except Exception as e:
            print_error(f"Error processing {doc_path}: {str(e)}")
    
    if not documents:
        print_error("No valid documents found.")
        return
    
    print_success(f"Loaded {len(documents)} documents")
    
    # Create FAISS index
    print("Creating FAISS vector store...")
    try:
        vector_store = FAISS.from_documents(documents, embedding_model)
        print_success("Vector store created successfully.")
        
        # Save the index if path is specified
        if args.save_index:
            vector_store.save_local(args.save_index)
            print_success(f"Vector index saved to {args.save_index}")
        
        return vector_store
    
    except Exception as e:
        print_error(f"Error creating FAISS vector store: {str(e)}")
        return None

def query_rag(args):
    """Query the RAG model with the specified question."""
    print_header("Querying RAG System")
    
    if not check_api_key():
        return
    
    # Load embedding model
    print("Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Load vector store
    print(f"Loading vector store from {args.index_path}...")
    try:
        vector_store = FAISS.load_local(args.index_path, embedding_model, allow_dangerous_deserialization=True)
        print_success("Vector store loaded successfully.")
    except Exception as e:
        print_error(f"Error loading vector store: {str(e)}")
        return
    
    # Initialize Gemini model
    print("Initializing LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.2,
    )
    
    # Create retriever
    print("Setting up retriever...")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': args.top_k if args.top_k else 2}
    )
    
    # Create RAG chain
    print("Creating RAG chain...")
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    # Run query
    print(f"\nQuestion: {Colors.BOLD}{args.query}{Colors.ENDC}")
    print("Searching and generating answer...")
    
    try:
        result = rag_chain.invoke({"query": args.query})
        
        # Print answer
        print("\n" + "=" * 50)
        print(f"{Colors.BLUE}{Colors.BOLD}Answer:{Colors.ENDC}")
        print(result["result"])
        print("=" * 50)
        
        # Print sources
        if args.show_sources and result.get("source_documents"):
            print(f"\n{Colors.BOLD}Sources:{Colors.ENDC}")
            for i, doc in enumerate(result["source_documents"]):
                print(f"\n{Colors.UNDERLINE}Source {i+1}:{Colors.ENDC}")
                print(f"  Content: {doc.page_content[:200]}...")
                print(f"  Metadata: {doc.metadata}")
    
    except Exception as e:
        print_error(f"Error during RAG chain execution: {str(e)}")

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="RAG-CLI: A user-friendly interface for G-RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process PDF command
    pdf_parser = subparsers.add_parser("process-pdf", help="Process a PDF file")
    pdf_parser.add_argument("input", help="Path to the PDF file")
    pdf_parser.add_argument("--output", "-o", help="Output directory (default: output/)")
    pdf_parser.add_argument("--tables", "-t", action="store_true", help="Extract tables from PDF")
    pdf_parser.add_argument("--markdown", "-m", action="store_true", help="Convert to markdown format")
    
    # Process Audio command
    audio_parser = subparsers.add_parser("process-audio", help="Process an audio file")
    audio_parser.add_argument("input", help="Path to the audio file")
    audio_parser.add_argument("--output", "-o", help="Output directory (default: output/audio/)")
    
    # Process Image command
    image_parser = subparsers.add_parser("process-image", help="Process an image file")
    image_parser.add_argument("input", help="Path to the image file")
    image_parser.add_argument("--output", "-o", help="Output directory (default: output/)")
    
    # Process Video command
    video_parser = subparsers.add_parser("process-video", help="Process a video file or YouTube URL")
    video_parser.add_argument("input", help="Path to video file or YouTube URL")
    video_parser.add_argument("--output", "-o", help="Output directory (default: output/video/)")
    
    # Create RAG command
    create_rag_parser = subparsers.add_parser("create-rag", help="Create a RAG model from documents")
    create_rag_parser.add_argument("documents", nargs="+", help="Paths to documents (txt, pdf, json)")
    create_rag_parser.add_argument("--save-index", "-s", help="Path to save the vector index")
    
    # Query RAG command
    query_rag_parser = subparsers.add_parser("query", help="Query a RAG model")
    query_rag_parser.add_argument("index_path", help="Path to the vector index")
    query_rag_parser.add_argument("query", help="Question to ask the RAG model")
    query_rag_parser.add_argument("--top-k", "-k", type=int, help="Number of documents to retrieve (default: 2)")
    query_rag_parser.add_argument("--show-sources", "-s", action="store_true", help="Show source documents")
    
    args = parser.parse_args()
    
    if args.command == "process-pdf":
        process_pdf(args)
    elif args.command == "process-audio":
        process_audio(args)
    elif args.command == "process-image":
        process_image(args)
    elif args.command == "process-video":
        process_video(args)
    elif args.command == "create-rag":
        create_rag(args)
    elif args.command == "query":
        query_rag(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()