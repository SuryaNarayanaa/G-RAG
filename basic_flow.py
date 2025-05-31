# ### AI model init 

# %%
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from dotenv import load_dotenv
load_dotenv()

# if "GOOGLE_API_KEY" not in os.environ:
#     os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


# Local Ollama Model 
from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage

# Replace this with your local Ollama model name
MODEL_NAME = "llama2"

# Initialize the Ollama chat model
# llm = ChatOllama(
#     model=MODEL_NAME,
#     base_url="http://localhost:11434",  # Default Ollama server URL
#     temperature=0,                   # Optional: Adjust sampling temperature
#     max_retries=3                      # Optional: Number of retries for failed requests
# )


# Example usage
# response = llm([HumanMessage(content="Explain the theory of relativity.")])
# print(response.content)


# %% [markdown]
# ### init custom LLMGraphTransformer and neo4j instance 

# %%
from src.graph_builder.llm import LLMGraphTransformer
llm_transformer = LLMGraphTransformer(llm=llm)



import os
from dotenv import load_dotenv
from src.graph_builder.lc_neo4j import Neo4jGraph
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "iambatman"
load_dotenv()
query_vector = llm_transformer._embed_text("")

graph = Neo4jGraph(refresh_schema=True,
                embedding_dim = len(query_vector)
)


# %% [markdown]
# ### Read document and split and convert to GraphDocuments

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1) Read your full text (e.g. from PDF)
full_text = open("op.txt", encoding="utf-8").read()

# 2) Wrap in a single Document
original_doc = Document(page_content=full_text, metadata={"doc_id":"mydoc"})

# 3) Configure the splitter
splitter = RecursiveCharacterTextSplitter(
    # target chunk size in characters (not tokens)
    chunk_size=500,
    # how many characters of overlap between chunks
    chunk_overlap=100,
    # what to split on first, second, etc.
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
)

# 4) Produce a list of Documents, each with its own chunk
chunked_docs = splitter.split_documents([original_doc])

# Inspect
for i, doc in enumerate(chunked_docs[:3]):
    print(f"--- chunk {i} ({len(doc.page_content)} chars) ---")
    print(doc.page_content[:200].replace("\n"," "), "…\n")
print(len(chunked_docs))


graph_documents = llm_transformer.convert_to_graph_documents(chunked_docs)


# %% [markdown]
# ### print pretty graph

# %%
def pretty_print_graph_documents(graph_documents):
    for idx, graph_doc in enumerate(graph_documents, start=1):
        print(f"\n### GraphDocument {idx}")
        print("**Nodes:**")
        for node_idx, node in enumerate(graph_doc.nodes, start=1):
            print(f"\n  {node_idx}. **Node ID:** {node.id}")
            print(f"     **Type:** {node.type}")
            print(f"     **Properties:**")
            for key, value in node.properties.items():
                if key == "source_texts":
                    print(f"       - **{key.capitalize()}:**")
                    for text in value:
                        print(f"         ```\n         {text}\n         ```")
                else:
                    print(f"       - **{key.capitalize()}:** {value}")

# Assuming `graph_documents` is the variable holding the output
pretty_print_graph_documents(graph_documents)
print(len(graph_documents))

# %% [markdown]
# ### Add GraphDocuments to DB

# %%
graph.setup_graph_with_documents(graph_documents)

# %%
from src.graph_builder.llm import LLMGraphTransformer
llm_transformer = LLMGraphTransformer(llm=llm)



import os
from dotenv import load_dotenv
from src.graph_builder.lc_neo4j import Neo4jGraph
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "iambatman"
load_dotenv()
query_vector = llm_transformer._embed_text("")

graph = Neo4jGraph(refresh_schema=True,
                embedding_dim = len(query_vector)
)


# %%
for doc in graph_documents:
    print(doc.relationships[0].properties['embedding'])   # Ensure embeddings are unique for each document
    print(doc.relationships[1].properties['embedding'])   # Ensure embeddings are unique for each document
    print(doc.relationships[2].properties['embedding'])   # Ensure embeddings are unique for each document


# %%
from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "iambatman"))

def get_all_node_embeddings(tx):
    query = "MATCH (n) WHERE n.embedding is NOT NULL RETURN id(n) AS id, n.embedding AS embedding, n.source_texts as source_texts"
    return list(tx.run(query))

def get_all_relationship_embeddings(tx):
    query = "MATCH ()-[r]->() WHERE r.embedding is NOT NULL RETURN id(r) AS id, r.embedding AS embedding,  r.source_texts as source_texts"
    return list(tx.run(query))

with driver.session() as session:
    node_results = session.read_transaction(get_all_node_embeddings)
    rel_results = session.read_transaction(get_all_relationship_embeddings)


# %%
len(node_results)

# %%
import numpy as np

# Suppose you have your query embedding as a numpy array
query_embedding = np.array([...])  # Your query embedding
query_text = "Surya Narayanaa N T"

query_embedding = llm_transformer._embed_text(query_text)

# Search across all node types
results = graph.search_similar_nodes(query_embedding)

# Or search relationships
rel_results = graph.search_similar_relationships(query_embedding,print_results=False)

# def cosine_similarity(a, b):
#     a = np.array(a)
#     b = np.array(b)
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# # For nodes
# node_similarities = []
# for record in node_results:
#     node_id = record["id"]
#     embedding = record["embedding"]
#     txt = record["source_texts"]
#     sim = cosine_similarity(query_embedding, embedding)
#     node_similarities.append((node_id, sim, txt))

# # For relationships
# rel_similarities = []
# for record in rel_results:
#     rel_id = record["id"]
#     embedding = record["embedding"]
#     sim = cosine_similarity(query_embedding, embedding)
#     rel_similarities.append((rel_id, sim))

# # Sort and get top-k
# top_k = 10
# top_nodes = sorted(node_similarities, key=lambda x: x[1], reverse=True)[:top_k]
# top_rels = sorted(rel_similarities, key=lambda x: x[1], reverse=True)[:top_k]


# %%
results 

# %%
# Function to print top k results
def print_top_k_results(results, k=5):
    """Print the top k results for each type from search results."""
    for type_name, type_results in results.items():
        print(f"\nTop {k} results for {type_name}:")
        if not type_results:
            print("  No results found")
            continue
            
        # Sort results by score in descending order
        sorted_results = sorted(type_results, key=lambda x: x['score'], reverse=True)
        for i, result in enumerate(sorted_results[:k]):
            item = result['item']
            
            # Get display property - prefer text/content/name/id
            display_props = ['text', 'content', 'name', 'id']
            display_value = next(
                (str(item.get(p)) for p in display_props if item.get(p) is not None),
                str(item.get('id', 'No ID'))
            )
            
            # Truncate long values
            if len(display_value) > 100:
                display_value = display_value[:97] + "..."
            
            # Print result with score
            print(f"  {i+1}. [{result['score']:.3f}] {display_value}")

def get_top_k_across_all_types(results, k=5):
    """Get the top k results across all node types combined.
    
    Args:
        results: Dict mapping node types to their search results
        k: Number of top results to return
        
    Returns:
        List of tuples (score, node_type, display_value)
    """
    all_results = []
    
    for type_name, type_results in results.items():
        if not type_results:
            continue
            
        for result in type_results:
            item = result['item']
            score = result['score']
            
            # Get display property - prefer text/content/name/id
            display_props = ['text', 'content', 'name', 'id', 'source_texts']
            display_value = next(
                (str(item.get(p)) for p in display_props if item.get(p) is not None),
                str(item.get('id', 'No ID'))
            )
            
            all_results.append((score, type_name, display_value))
    
    # Sort by score in descending order and get top k
    top_k_results = sorted(all_results, key=lambda x: x[0], reverse=True)[:k]
    
    # Print formatted results
    print("\nTop", k, "results across all types:")
    for score, type_name, display_value in top_k_results:
        # Truncate long values
        if len(display_value) > 100:
            display_value = display_value[:97] + "..."
        print(f"  [{score:.3f}] ({type_name}) {display_value}")
        
    return top_k_results

# Get the top 5 results across all types
# top_5_overall = get_top_k_across_all_types(results, k=5)

# # Create a formatted string that can be passed to an LLM
# llm_input = "Top 5 most relevant results:\n" + "\n".join([
#     f"{i+1}. [{score:.3f}] ({type_name}) {value}" 
#     for i, (score, type_name, value) in enumerate(top_5_overall)
# ])

# %%
results

# %%
def get_top_k_with_sources(results, k=5):
    """Get the top k results across all node types with their source information.
    
    Args:
        results: Dict mapping node types to their search results
        k: Number of top results to return
        
    Returns:
        List of tuples (score, node_type, display_value, source_texts)
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
            
            # Get source information if available
            source_texts = item.get('source_texts', [])
            if not isinstance(source_texts, list):
                source_texts = [str(source_texts)]
            
            all_results.append((score, type_name, display_value, source_texts))
    
    # Sort by score in descending order and get top k
    top_k_results = sorted(all_results, key=lambda x: x[0], reverse=True)[:k]
    
    # Print formatted results
    print("\nTop", k, "results across all types:")
    for score, type_name, display_value, source_texts in top_k_results:
        print(f"\n[{score:.3f}] ({type_name}) {display_value}")
        if source_texts:
            print("Source:")
            for text in source_texts:
                print(textwrap.fill(text, width=100, initial_indent="  ", subsequent_indent="  "))
        
    return top_k_results

# Import textwrap for pretty printing
import textwrap

# Get the top 5 results across all types with sources
top_5_with_sources = get_top_k_with_sources(results, k=5)

# Create a formatted string for LLM use
llm_input = "Top 5 most relevant results with sources:\n\n" + "\n\n".join([
    f"{i+1}. [{score:.3f}] ({type_name}) {value}\nSource:\n" + "\n".join(f"  {text}" for text in sources)
    for i, (score, type_name, value, sources) in enumerate(top_5_with_sources)
])

# %%
def format_rag_prompt(results, user_query):
    """Format search results and create a structured prompt for Gemini.
    
    Args:
        results: Dict mapping node types to their search results
        user_query: The original user query
        
    Returns:
        Tuple[str, str]: System prompt and formatted context
    """
    # First get top results with sources
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
    top_results = sorted(all_results, key=lambda x: x[0], reverse=True)[:5]
    
    # Create system prompt
    system_prompt = f"""You are a specialized RAG (Retrieval Augmented Generation) agent designed to:
1. Answer questions using ONLY the provided context
2. Maintain academic accuracy and precision
3. Cite your sources when providing information
4. Admit when you don't have enough information to answer
5. Focus on the most relevant information from the highest-scoring matches
6. Consider both the content and the type of each retrieved item

USER QUERY : {user_query}
REMEMBER THIS WELL: Analyze the user query first, if it has some general data that is not provided in the context of retrieved item, try to repond to the user GENERALLY 

Base your responses ONLY on the retrieved context provided below. """

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

    return system_prompt, context

# Import required modules
import textwrap
user_query = "   Dr. Karpagam G R1, Adhish Krishna S2, Mohana Kumar P3, Sanjay J4, Surya Narayanaa N T5"
# Get the search results formatted for RAG
system_prompt, context = format_rag_prompt(results, user_query)

# Format the complete prompt for Gemini
gemini_input = f"""{system_prompt}

{context}

User Query: {user_query}

Please provide a detailed answer based on the retrieved information above."""

print("Generated prompt for Gemini:")
print("-" * 80)
print(gemini_input)
print("-" * 80)

# The gemini_input variable is now ready to be sent to the Gemini API

# %%
def query_gemini(prompt):
    """Send a prompt to Gemini API and get the response.
    
    Args:
        prompt: The formatted prompt to send to Gemini
        
    Returns:
        str: Gemini's response
    """
    # We already have ChatGoogleGenerativeAI initialized as 'llm'
    # No need to reinitialize the client
    try:
        # Use the existing llm instance to generate response
        response = llm.invoke(prompt)
        
        # Return the response content
        return response.content
        
    except Exception as e:
        return f"Error querying Gemini: {str(e)}"

# Test the function with the existing gemini_input
try:
    response = query_gemini(gemini_input)
    
    print("\nGemini's Response:")
    print("-" * 80)
    print(response)
    print("-" * 80)
except Exception as e:
    print(f"Error: {str(e)}")


# %%



