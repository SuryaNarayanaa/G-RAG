import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import textwrap
import logging
from termcolor import colored

# Import ingestion components
from src.ingestion.connectors.pdf_parser import PDFParser
from src.ingestion.connectors.audio_parser import AudioParser
from src.ingestion.connectors.image_parser import ImageParser
from src.ingestion.connectors.video_parser import VideoParser

# Import RAG components
from langchain_google_genai import ChatGoogleGenerativeAI
from src.graph_builder.llm import LLMGraphTransformer
from src.graph_builder.lc_neo4j import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chat_models import ChatOllama

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IntegratedGraphRAG:
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "iambatman",
        google_api_key: Optional[str] = None,
        output_dir: str = "output"
    ):
        """Initialize the integrated Graph RAG system.
        
        Args:
            neo4j_uri: URI for Neo4j database
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            google_api_key: Google API key for Gemini
            output_dir: Directory to store processed files
        """
        # Load environment variables
        load_dotenv()
        
        # Set up environment variables
        os.environ["NEO4J_URI"] = neo4j_uri
        os.environ["NEO4J_USERNAME"] = neo4j_username
        os.environ["NEO4J_PASSWORD"] = neo4j_password
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
            
        # Replace this with your local Ollama model name
        MODEL_NAME = "llama2"

        # Initialize the Ollama chat model
        # self.llm = ChatOllama(
        #     model=MODEL_NAME,
        #     base_url="http://localhost:11434",  # Default Ollama server URL
        #     temperature=0,                   # Optional: Adjust sampling temperature
        #     max_retries=3                      # Optional: Number of retries for failed requests
        # )
        self.llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

                    
        # Initialize LLM transformer
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        
        # Initialize parsers
        self.pdf_parser = PDFParser
        self.audio_parser = AudioParser
        self.image_parser = ImageParser
        self.video_parser = VideoParser
        
        # Set output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Get embedding dimension for Neo4j setup
        query_vector = self.llm_transformer._embed_text("")
        
        # Initialize Neo4j
        self.graph = Neo4jGraph(
            refresh_schema=True,
            embedding_dim=len(query_vector)
        )
        logging.info(colored("Neo4j graph initialized.", "green"))

    def process_document(self, file_path: str, file_type: Optional[str] = None) -> None:
        """Process a document and add it to the graph.
        
        Args:
            file_path: Path to the file to process
            file_type: Type of file ('pdf', 'audio', 'video', 'image'). If None, will try to infer from extension.
        """
        logging.info(colored(f"Processing document: {file_path}", "blue"))
        if file_type is None:
            ext = os.path.splitext(file_path)[1].lower()
            file_type = ext[1:] if ext else None
            logging.info(colored(f"Inferred file type: {file_type}", "blue"))
            
        # Extract text based on file type
        if file_type == 'pdf':
            parser = self.pdf_parser(file_path, output_dir=self.output_dir)
            text_file, _ = parser.extract_text_and_images()
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            logging.info(colored(f"Extracted text from PDF: {file_path}", "blue"))
        
        elif file_type in ['wav', 'mp3', 'audio']:
            parser = self.audio_parser(file_path)
            wav_path = os.path.join(self.output_dir, f"{os.path.basename(file_path)}.wav")
            self.audio_parser.convert_to_wav(file_path, wav_path)
            transcript_file = parser.extract_text_with_whisper(output_dir=self.output_dir)
            with open(transcript_file, 'r', encoding='utf-8') as f:
                text = f.read()
            logging.info(colored(f"Extracted text from audio: {file_path}", "blue"))
        
        elif file_type in ['mp4', 'video']:
            parser = self.video_parser(file_path)
            text = parser.extract_text()
            logging.info(colored(f"Extracted text from video: {file_path}", "blue"))
        
        elif file_type in ['jpg', 'png', 'image']:
            parser = self.image_parser(file_path)
            text = parser.extract_text()
            logging.info(colored(f"Extracted text from image: {file_path}", "blue"))
        
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Create document
        doc = Document(page_content=text, metadata={"doc_id": os.path.basename(file_path)})
        
        # Split into chunks
        chunked_docs = self.text_splitter.split_documents([doc])
        logging.info(colored(f"Split document into {len(chunked_docs)} chunks.", "blue"))
        
        # Convert to graph documents
        graph_documents = self.llm_transformer.convert_to_graph_documents(chunked_docs)
        logging.info(colored("Converted documents to graph documents.", "blue"))
        
        # Add to graph database
        self.graph.setup_graph_with_documents(graph_documents)
        
        print(f"Successfully processed {file_path} and added to graph database")
        logging.info(colored(f"Successfully processed {file_path} and added to graph database", "green"))

    def query(self, query_text: str, k: int = 5) -> str:
        """Query the graph RAG system.
        
        Args:
            query_text: The query text
            k: Number of top results to return
            
        Returns:
            str: Response from the RAG system
        """
        logging.info(colored(f"Received query: {query_text}", "yellow"))
        # Get query embedding
        query_embedding = self.llm_transformer._embed_text(query_text)
        
        # Search nodes
        results = self.graph.search_similar_nodes(query_embedding)
        logging.info(colored(f"Found {len(results)} results.", "yellow"))
        
        # Format RAG prompt
        system_prompt = f"""You are a specialized RAG (Retrieval Augmented Generation) agent designed to:
1. Answer questions using ONLY the provided context
2. Maintain academic accuracy and precision
3. Cite your sources when providing information
4. Admit when you don't have enough information to answer
5. Focus on the most relevant information from the highest-scoring matches
6. Consider both the content and the type of each retrieved item

USER QUERY: {query_text}
REMEMBER THIS WELL: Analyze the user query first, if it has some general data that is not provided in the context of retrieved item, try to respond to the user GENERALLY 

Base your responses ONLY on the retrieved context provided below."""

        # Get context from results
        context = self._format_search_results(results, k)
        
        # Create complete prompt
        prompt = f"{system_prompt}\n\n{context}\n\nUser Query: {query_text}\n\nPlease provide a detailed answer based on the retrieved information above."
        
        # Get response from LLM
        response = self.llm.invoke(prompt)
        logging.info(colored("Received response from LLM.", "yellow"))
        
        return response.content

    def _format_search_results(self, results: Dict[str, List[Dict[str, Any]]], k: int = 5) -> str:
        """Format search results for the LLM.
        
        Args:
            results: Search results from Neo4j
            k: Number of top results to include
            
        Returns:
            str: Formatted context
        """
        all_results = []
        
        for type_name, type_results in results.items():
            if not type_results:
                continue
                
            for result in type_results:
                item = result['item']
                score = result['score']
                
                # Get basic properties
                display_props = ['text', 'content', 'name', 'id']
                display_value = next(
                    (str(item.get(p)) for p in display_props if item.get(p) is not None),
                    str(item.get('id', 'No ID'))
                )
                
                # Get source information
                source_texts = item.get('source_texts', [])
                if not isinstance(source_texts, list):
                    source_texts = [str(source_texts)]
                
                all_results.append((score, type_name, display_value, source_texts))
        
        # Sort by score and get top results
        top_results = sorted(all_results, key=lambda x: x[0], reverse=True)[:k]
        
        # Format context
        context = "Retrieved Context (sorted by relevance):\n\n"
        for i, (score, type_name, value, sources) in enumerate(top_results):
            context += f"[Result {i+1}] Score: {score:.3f} | Type: {type_name}\n"
            context += f"Item: {value}\n"
            if sources:
                context += "Source:\n"
                for text in sources:
                    context += textwrap.fill(text, width=100, initial_indent="  ", subsequent_indent="  ") + "\n"
            context += "\n"
            
        return context

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'graph'):
            self.graph.close()
            logging.info(colored("Neo4j graph connection closed.", "red"))
