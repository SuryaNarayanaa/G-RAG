"""
    ✅- llm          
    ✅- graph driver 
    ✅- graph schemas (GraphDocument, Node, Relationship)
    ❌- 
        self._function_call = not ignore_tool_usage
search above to continue 

"""
# graph.add_graph_documents(graph_documents, include_source=True, baseEntityLabel=True)

from langchain_neo4j import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.base_language import BaseLanguageModel

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from typing import List, Optional, Union, Tuple
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

from dotenv import load_dotenv

load_dotenv()

graph = Neo4jGraph(refresh_schema=True)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

class G_RAG:
    def __init__(
        self, 
        llm: BaseLanguageModel,
        allowed_nodes: List[str] = [], # CORE_NODE_TYPES by default?
        allowed_relationships: Union[List[str], List[Tuple[str, str, str]]] = [], # CORE_RELATIONSHIP_TYPES?
        prompt: Optional[ChatPromptTemplate] = None,
    ):
        

