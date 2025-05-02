# RAG-CLI: User Guide

## Introduction

RAG-CLI is a user-friendly command line interface for the G-RAG (Generative Retrieval-Augmented Generation) system. It provides a simple way to process various types of media files (PDF, audio, images, and videos) and build powerful Retrieval-Augmented Generation systems with minimal effort.

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Commands](#commands)
  - [Processing Files](#processing-files)
  - [Creating RAG Models](#creating-rag-models)
  - [Querying RAG Models](#querying-rag-models)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Installation

RAG-CLI requires Python 3.8+ and several dependencies. To set up:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/G-RAG.git
   cd G-RAG
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root directory and add:
   ```
   GOOGLE_API_KEY=your_google_ai_api_key
   ```
   You can obtain an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Getting Started

RAG-CLI can be run directly from the command line:

```bash
python rag-cli/rag_cli.py [command] [options]
```

For help:

```bash
python rag-cli/rag_cli.py --help
```

## Commands

### Processing Files

#### Process PDF Documents

Extract text, images, and tables from PDF documents.

```bash
python rag-cli/rag_cli.py process-pdf <path_to_pdf> [options]
```

**Options:**
- `--output`, `-o`: Output directory (default: output/)
- `--tables`, `-t`: Extract tables from PDF
- `--markdown`, `-m`: Convert to markdown format

#### Process Audio Files

Transcribe audio files using Whisper.

```bash
python rag-cli/rag_cli.py process-audio <path_to_audio> [options]
```

**Options:**
- `--output`, `-o`: Output directory (default: output/audio/)

#### Process Image Files

Extract text and metadata from images.

```bash
python rag-cli/rag_cli.py process-image <path_to_image> [options]
```

**Options:**
- `--output`, `-o`: Output directory (default: output/)

#### Process Video Files or YouTube URLs

Download YouTube videos or process local video files and extract transcripts.

```bash
python rag-cli/rag_cli.py process-video <path_or_url> [options]
```

**Options:**
- `--output`, `-o`: Output directory (default: output/video/)

### Creating RAG Models

Create a RAG model from multiple document sources.

```bash
python rag-cli/rag_cli.py create-rag <documents...> --save-index <path>
```

**Options:**
- `--save-index`, `-s`: Path to save the vector index

### Querying RAG Models

Query a RAG model with a question.

```bash
python rag-cli/rag_cli.py query <index_path> "<your_question>" [options]
```

**Options:**
- `--top-k`, `-k`: Number of documents to retrieve (default: 2)
- `--show-sources`, `-s`: Show source documents

## Examples

### Processing a PDF with Tables and Converting to Markdown

```bash
python rag-cli/rag_cli.py process-pdf testing/1.pdf --tables --markdown
```

This command will:
1. Extract text from the PDF
2. Extract all images from the PDF
3. Extract tables as CSV files
4. Convert the PDF to markdown format

### Processing a YouTube Video

```bash
python rag-cli/rag_cli.py process-video "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

This command will:
1. Download the YouTube video
2. Extract audio from the video
3. Transcribe the audio using Whisper
4. Save the transcript to the output directory

### Creating a RAG Model from Multiple Documents

```bash
python rag-cli/rag_cli.py create-rag output/op.txt output/audio/transcript_whisper.txt --save-index indexes/my_knowledge_base
```

This command will:
1. Process the specified documents
2. Create embeddings using a language model
3. Build a FAISS vector index
4. Save the index to the specified location

### Querying a RAG Model

```bash
python rag-cli/rag_cli.py query indexes/my_knowledge_base "What is Neuro Prune?" --show-sources
```

This command will:
1. Load the specified index
2. Process the question
3. Retrieve relevant documents
4. Generate an answer using the LLM
5. Display the answer and source documents

## Troubleshooting

### API Key Issues

If you encounter API key errors:

1. Make sure your `.env` file contains the correct API key
2. Ensure the `.env` file is in the project root directory
3. Try setting the API key directly in the environment:
   ```bash
   export GOOGLE_API_KEY=your_api_key  # Linux/macOS
   set GOOGLE_API_KEY=your_api_key     # Windows
   ```

### Processing Errors

If file processing fails:

1. Check if the file exists and is accessible
2. Ensure the file format is supported
3. Check the console output for specific error messages

### Vector Store Errors

If you encounter issues with the FAISS vector store:

1. Ensure you have installed all required dependencies
2. Check if you have sufficient disk space for index creation
3. Try using a smaller batch of documents for initial testing

## Contributing

Contributions to RAG-CLI are welcome! Please feel free to submit pull requests or open issues to improve the tool.

## License

This project is licensed under the MIT License - see the LICENSE file for details.