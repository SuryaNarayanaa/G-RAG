"""
    ✅- llm          
    ✅- graph driver 
    ✅- graph schemas (GraphDocument, Node, Relationship)
    ❌- self._function_call = not ignore_tool_usage
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
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser


from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
#-------------------------INSTANCES--------------------------#
graph = Neo4jGraph(refresh_schema=True)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
#-------------------------INSTANCES--------------------------#


# ---------- Define Core Generic Types for Generalized Extraction ----------#
CORE_NODE_TYPES = ["Entity", "Concept", "Event", "Location", "Organization", "Person", "Time", "Document", "Property", "Condition", "Other"]

examples = [
    {
        "text": (
            "Adam is a software engineer in Microsoft since 2009, "
            "and last year he got an award as the Best Talent"
        ),
        "head": "Adam",
        "head_type": "Person", # Use CORE type
        "relation": "WORKS_FOR", # This is specific, could be debated vs. INVOLVED_IN
        "tail": "Microsoft",
        "tail_type": "Organization", # Use CORE type
    },
    {
        "text": (
            "Adam is a software engineer in Microsoft since 2009, "
            "and last year he got an award as the Best Talent"
        ),
        "head": "Adam",
        "head_type": "Person",
        "relation": "HAS_ATTRIBUTE", # Changed from HAS_AWARD
        "tail": "Best Talent award", # Slightly rephrased tail for clarity
        "tail_type": "Property", # Changed from Award
    },
    {
        "text": (
            "Microsoft is a tech company that provide "
            "several products such as Microsoft Word"
        ),
        "head": "Microsoft Word",
        "head_type": "Entity", # Changed from Product
        "relation": "CREATED", # Changed from PRODUCED_BY
        "tail": "Microsoft",
        "tail_type": "Organization",
    },
    {
        "head": "Microsoft Word",
        "head_type": "Entity",
        "relation": "HAS_ATTRIBUTE", # Changed from HAS_CHARACTERISTIC
        "tail": "lightweight app",
        "tail_type": "Property", # Changed from Characteristic
    },
    {
        "text": "Microsoft Word is a lightweight app that accessible offline",
        "head": "Microsoft Word",
        "head_type": "Entity",
        "relation": "HAS_ATTRIBUTE", # Changed
        "tail": "accessible offline",
        "tail_type": "Property", # Changed
    },
]

CORE_RELATIONSHIP_TYPES = [
    "RELATED_TO",       # Generic fallback
    "HAS_PROPERTY",     # Connects an entity to its property node
    "ATTRIBUTE_OF",     # Connects a property node back to its entity (can be redundant w/ HAS_PROPERTY)
    "PART_OF",          # Composition/aggregation
    "LOCATED_IN",       # Spatial relationship
    "OCCURRED_AT",      # Event location/time
    "OCCURRED_ON",      # Event time
    "INVOLVED_IN",      # Participation in an event/process
    "MENTIONS",         # Document mentioning an entity/concept
    "CREATED",          # Creation relationship
    "HAS_ATTRIBUTE",    # Similar to HAS_PROPERTY, potentially more abstract
    "IS_A",             # Subclass/instance relationship (use sparingly)
    "EXAMPLE_OF",       # Specific example of a concept
    "CAUSES",           # Causal relationship
    "AFFECTS",          # Influence relationship
    "MEASURES",         # Relating a property/metric to what it measures
    "CONTAINS",         # General containment
    "EQUIVALENT_TO",    # Semantic equivalence
]

#tuples for more constraint (Optional, use if specific patterns are critical)
CORE_RELATIONSHIP_TUPLES = [
    ("Person", "WORKS_FOR", "Organization"),
    ("Event", "OCCURRED_AT", "Location"),
    ("Event", "OCCURRED_ON", "Time"),
    ("Entity", "HAS_PROPERTY", "Property"),
    ("Document", "MENTIONS", "Entity"),
]

system_prompt = (
    "# Knowledge Graph Instructions\n"
    "## 1. Overview\n"
    "You are a specialized algorithm for extracting structured information to build a knowledge graph.\n"
    "Focus on accuracy. Do not add information not explicitly mentioned in the text.\n"
    "Extract nodes and relationships relevant to the input text.\n"
    "- **Nodes** represent entities, concepts, events, properties, etc.\n"
    "- **Relationships** represent connections between nodes.\n"
    "The goal is a clear and concise knowledge graph.\n"
    "## 2. Node Labeling\n"
    f"- **Use Core Types**: Preferentially use the following node types: {', '.join(CORE_NODE_TYPES)}.\n"
    "- **Consistency**: Use the same type for the same conceptual entity (e.g., always 'Person' for individuals).\n"
    "- **Node IDs**: Use human-readable names or identifiers found *directly* in the text. Avoid integers or generated IDs.\n"
    "- **Specificity**: Only use a non-core type if *absolutely necessary* and clearly supported by the text.\n"
    "## 3. Relationship Labeling\n"
    f"- **Use Core Types**: Preferentially use the following relationship types: {', '.join(CORE_RELATIONSHIP_TYPES)}.\n"
    "- **Generality**: Use general, timeless relationship types (e.g., 'LOCATED_IN' instead of 'MOVED_TO').\n"
    "- **Clarity**: Ensure the relationship type accurately reflects the connection described in the text.\n"
    "- **Direction**: Ensure the subject->object direction is correct for the relationship type.\n"
    "## 4. Coreference Resolution\n"
    "- **Maintain Entity Consistency**: If an entity (e.g., \"Dr. Jane Smith\") is referred to later by variations (\"Smith\", \"Jane\", \"she\"), always use the most complete identifier (\"Dr. Jane Smith\") as the node ID throughout the graph for that entity.\n"
    "## 5. Property Extraction (If Applicable)\n"
    "- If extracting node or relationship properties, ensure the 'key' clearly identifies the property and the 'value' is the corresponding information from the text. Format dates as yyyy-mm-dd.\n"
    "## 6. Strict Compliance\n"
    "- Adhere strictly to these rules. Output *only* the requested structured data (nodes and relationships)."
)
#---------- Define Core Generic Types for Generalized Extraction ----------#

#-------------------------Pydantic Schemas--------------------------#

class UnstructuredRelation(BaseModel):
    head: str = Field(
        description=(
            "extracted head entity like Microsoft, Apple, John. "
            "Must use human-readable unique identifier."
        )
    )
    head_type: str = Field(
        description="type of the extracted head entity like Person, Company, etc"
    )
    relation: str = Field(description="relation between the head and the tail entities")
    tail: str = Field(
        description=(
            "extracted tail entity like Microsoft, Apple, John. "
            "Must use human-readable unique identifier."
        )
    )
    tail_type: str = Field(
        description="type of the extracted tail entity like Person, Company, etc"
    )
    
#-------------------------Pydantic Schemas--------------------------#


#----------------------HELPER FUNCTIONS----------------------#

def validate_and_get_relationship_type(
    allowed_relationships: Union[List[str], List[Tuple[str, str, str]]],
    allowed_nodes: Optional[List[str]],
) -> Optional[str]:
    """Validates the format of allowed relationships and returns its type ('string' or 'tuple').

    This function checks if the `allowed_relationships` argument conforms to one of
    the expected formats: either a list of strings (relationship type names) or a
    list of 3-element tuples (source_node_type, relationship_type, target_node_type).
    If the tuple format is used, it also validates that the source and target node
    types exist within the `allowed_nodes` list.

    Args:
        allowed_relationships (Union[List[str], List[Tuple[str, str, str]]]):
            Specifies the allowed relationships. Can be an empty list, a list of
            relationship type strings, or a list of 3-element tuples representing
            (source_type, relationship_type, target_type).
        allowed_nodes (Optional[List[str]]):
            A list of node type names. This is required and used for validation
            only if `allowed_relationships` is provided as a list of tuples.
            The first and third elements of each tuple must be present in this list.

    Raises:
        ValueError: If `allowed_relationships` is provided but is not a list.
        ValueError: If `allowed_relationships` is a list but contains invalid items
            (e.g., mixed types, tuples of wrong size, or tuples with source/target
            nodes not found in `allowed_nodes` when `allowed_nodes` is provided).

    Returns:
        Optional[str]:
            - "string" if `allowed_relationships` is a valid list of strings.
            - "tuple" if `allowed_relationships` is a valid list of 3-element tuples
                adhering to `allowed_nodes` constraints.
            - None if `allowed_relationships` is an empty list.
    """
    if allowed_relationships and not isinstance(allowed_relationships, list):
        raise ValueError("`allowed_relationships` attribute must be a list.")

    # If it's an empty list
    if not allowed_relationships:
        return None

    # Validate list of strings
    if all(isinstance(item, str) for item in allowed_relationships):
        return "string"  # Valid: all items are strings, no further checks needed.

    # Validate list of 3-tuples and check if first/last elements are in allowed_nodes
    # Note: allowed_nodes check only happens if allowed_nodes is not None/empty
    if allowed_nodes and all(
        isinstance(item, tuple)
        and len(item) == 3
        and all(isinstance(subitem, str) for subitem in item)
        and item[0] in allowed_nodes  # Check source node type
        and item[2] in allowed_nodes  # Check target node type
        for item in allowed_relationships
    ):
        return "tuple"  # all items are 3-tuples, and the first/last elements are in allowed_nodes.

    # Handle case where allowed_relationships is tuples but allowed_nodes is None/empty
    # This case should likely raise an error or be handled based on requirements,
    # but current logic proceeds to the final raise.
    # If we are here, it might also be because allowed_relationships is tuples but allowed_nodes is None
    # or because it's not purely strings or purely valid tuples.

    # If the input doesn't match any of the valid cases, raise a ValueError
    raise ValueError(
        "`allowed_relationships` must be list of strings or a list of 3-item tuples. "
        "For tuples, the first and last elements must be in the `allowed_nodes` list (if provided)."
    )


def create_unstructured_prompt(
    node_labels: Optional[List[str]] = None, # Suggest passing CORE_NODE_TYPES here
    rel_types: Optional[Union[List[str], List[Tuple[str, str, str]]]] = None, # Suggest passing CORE_RELATIONSHIP_TYPES here
    relationship_type: Optional[str] = None,
    additional_instructions: Optional[str] = "",
) -> ChatPromptTemplate:

    node_labels_str = str(node_labels) if node_labels else f"general types like {CORE_NODE_TYPES}"
    if rel_types:
        if relationship_type == "tuple":
            rel_types_str = str(list({item[1] for item in rel_types}))
        else:
            rel_types_str = str(rel_types)
    else:
        rel_types_str = f"general types like {CORE_RELATIONSHIP_TYPES}"

    base_string_parts = [
        "You are a top-tier algorithm designed for extracting information in "
        "structured formats to build a knowledge graph. Your task is to identify "
        "the entities and relations requested with the user prompt from a given "
        "text. You must generate the output in a JSON format containing a list "
        'with JSON objects. Each object should have the keys: "head", '
        '"head_type", "relation", "tail", and "tail_type".',
        # Updated guidance below
        'The "head" key must contain the text of the extracted head entity.',
        f'The "head_type" key must contain the type of the extracted head entity. '
        f"Prefer types from this list: {node_labels_str}.",
        f'The "relation" key must contain the type of relation between the "head" '
        f'and the "tail". Prefer relation types from this list: {rel_types_str}. Use general, timeless types.',
        'The "tail" key must represent the text of an extracted entity which is '
        'the tail of the relation.',
        f'The "tail_type" key must contain the type '
        f"of the tail entity. Prefer types from this list: {node_labels_str}.",
        # End updated guidance
        "Your task is to extract relationships from text strictly adhering "
        "to the provided schema. The relationships can only appear "
        "between specific node types are presented in the schema format "
        "like: (Entity1Type, RELATIONSHIP_TYPE, Entity2Type) /n"
        f"Provided schema is {rel_types}"
        if relationship_type == "tuple"
        else "",
        "Attempt to extract as many entities and relations as you can. Maintain "
        "Entity Consistency: When extracting entities, it's vital to ensure "
        'consistency. If an entity, such as "John Doe", is mentioned multiple '
        "times in the text but is referred to by different names or pronouns "
        '(e.g., "Joe", "he"), always use the most complete identifier for '
        "that entity (e.g. \"John Doe\").",
        "IMPORTANT NOTES:\n- Output *only* the JSON list of objects.",
        "- Do not add any explanations or introductory text.",
        additional_instructions,
    ]
    system_prompt = "\n".join(filter(None, base_string_parts))
    system_message = SystemMessage(content=system_prompt)
    parser = JsonOutputParser(pydantic_object=UnstructuredRelation)

    human_string_parts = [
        
        "Based on the following example, extract entities and "
        "relations from the provided text.\n\n",
        "Use the following entity types:"
        f"# ENTITY TYPES: {node_labels_str}\n"
        if node_labels # Check if specific labels were actually passed
        else f"# ENTITY TYPES: Prefer general types like {CORE_NODE_TYPES}\n",
        "Use the following relation types:"
        f"# RELATION TYPES: {rel_types_str}\n"
        if rel_types # Check if specific relations were actually passed
        else f"# RELATION TYPES: Prefer general types like {CORE_RELATIONSHIP_TYPES}\n",
        "Your task is to extract relationships from text strictly adhering "
        "to the provided schema. The relationships can only appear "
        "between specific node types are presented in the schema format "
        "like: (Entity1Type, RELATIONSHIP_TYPE, Entity2Type) /n"
        f"Provided schema is {rel_types}"
        
        if relationship_type == "tuple"

        else "",
        
        "Below are a number of examples of text and their extracted "
        "entities and relationships."
        "{examples}\n",
        additional_instructions,
        "For the following text, extract entities and relations as "
        "in the provided example."
        "{format_instructions}\nText: {input}",
    ]
    human_prompt_string = "\n".join(filter(None, human_string_parts))
    #TODO: Change the below prompt and change the examples
    human_prompt = PromptTemplate(
        template=human_prompt_string,
        input_variables=["input"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            # Pass the strings created above for the prompt template
            "node_labels": node_labels_str if node_labels else f"Prefer general types like {CORE_NODE_TYPES}",
            "rel_types": rel_types_str if rel_types else f"Prefer general types like {CORE_RELATIONSHIP_TYPES}",
            "examples": examples,
        },
    )
    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message, human_message_prompt]
    )
    return chat_prompt

def get_default_prompt(
    additional_instructions: str = "",
) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt), #TODO change the system prompt 
            (
                "human",
                additional_instructions
                + " Tip: Make sure to answer in the correct format and do "
                "not include any explanations. "
                "Use the given format to extract information from the "
                "following input: {input}",
            ),
        ]
    )

#TODO: Complete the below fn
def create_simple_model(
    node_labels: Optional[List[str]] = None,
    rel_types: Optional[Union[List[str], List[Tuple[str, str, str]]]] = None,
    node_properties: Union[bool, List[str]] = False,
    llm_type: Optional[str] = None,
    relationship_properties: Union[bool, List[str]] = False,
    relationship_type: Optional[str] = None,
) -> Type[GraphDocument]:
    """
    Create a simple graph model with optional constraints on node
    and relationship types.

    Args:
        node_labels (Optional[List[str]]): Specifies the allowed node types.
            Defaults to None, allowing all node types.
        rel_types (Optional[List[str]]): Specifies the allowed relationship types.
            Defaults to None, allowing all relationship types.
        node_properties (Union[bool, List[str]]): Specifies if node properties should
            be included. If a list is provided, only properties with keys in the list
            will be included. If True, all properties are included. Defaults to False.
        relationship_properties (Union[bool, List[str]]): Specifies if relationship
            properties should be included. If a list is provided, only properties with
            keys in the list will be included. If True, all properties are included.
            Defaults to False.
        llm_type (Optional[str]): The type of the language model. Defaults to None.
            Only openai supports enum param: openai-chat.

    Returns:
        Type[_Graph]: A graph model with the specified constraints.

    Raises:
        ValueError: If 'id' is included in the node or relationship properties list.
    """

        : Dict[str, Tuple[Any, Any]] = {
        "id": (
            str,
            Field(..., description="Name or human-readable unique identifier."),
        ),
        "type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
    }

    if node_properties:
        if isinstance(node_properties, list) and "id" in node_properties:
            raise ValueError("The node property 'id' is reserved and cannot be used.")
        # Map True to empty array
        node_properties_mapped: List[str] = (
            [] if node_properties is True else node_properties
        )

        class Property(BaseModel):
            """A single property consisting of key and value"""

            key: str = optional_enum_field(
                node_properties_mapped,
                description="Property key.",
                input_type="property",
                llm_type=llm_type,
            )
            value: str = Field(
                ...,
                description=(
                    "Extracted value. Any date value "
                    "should be formatted as yyyy-mm-dd."
                ),
            )

        node_fields["properties"] = (
            Optional[List[Property]],
            Field(None, description="List of node properties"),
        )
    SimpleNode = create_model("SimpleNode", **node_fields)  # type: ignore

    relationship_fields: Dict[str, Tuple[Any, Any]] = {
        "source_node_id": (
            str,
            Field(
                ...,
                description="Name or human-readable unique identifier of source node",
            ),
        ),
        "source_node_type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the source node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
        "target_node_id": (
            str,
            Field(
                ...,
                description="Name or human-readable unique identifier of target node",
            ),
        ),
        "target_node_type": (
            str,
            optional_enum_field(
                node_labels,
                description="The type or label of the target node.",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
        "type": (
            str,
            optional_enum_field(
                rel_types,
                description="The type of the relationship.",
                input_type="relationship",
                llm_type=llm_type,
                relationship_type=relationship_type,
            ),
        ),
    }
    if relationship_properties:
        if (
            isinstance(relationship_properties, list)
            and "id" in relationship_properties
        ):
            raise ValueError(
                "The relationship property 'id' is reserved and cannot be used."
            )
        # Map True to empty array
        relationship_properties_mapped: List[str] = (
            [] if relationship_properties is True else relationship_properties
        )

        class RelationshipProperty(BaseModel):
            """A single property consisting of key and value"""

            key: str = optional_enum_field(
                relationship_properties_mapped,
                description="Property key.",
                input_type="property",
                llm_type=llm_type,
            )
            value: str = Field(
                ...,
                description=(
                    "Extracted value. Any date value "
                    "should be formatted as yyyy-mm-dd."
                ),
            )

        relationship_fields["properties"] = (
            Optional[List[RelationshipProperty]],
            Field(None, description="List of relationship properties"),
        )
    SimpleRelationship = create_model("SimpleRelationship", **relationship_fields)  # type: ignore
    # Add a docstring to the dynamically created model
    if relationship_type == "tuple":
        SimpleRelationship.__doc__ = (
            "Your task is to extract relationships from text strictly adhering "
            "to the provided schema. The relationships can only appear "
            "between specific node types are presented in the schema format "
            "like: (Entity1Type, RELATIONSHIP_TYPE, Entity2Type) /n"
            f"Provided schema is {rel_types}"
        )

    class DynamicGraph(GraphDocument):
        """Represents a graph document consisting of nodes and relationships."""

        nodes: Optional[List[SimpleNode]] = Field(description="List of nodes")  # type: ignore
        relationships: Optional[List[SimpleRelationship]] = Field(  # type: ignore
            description="List of relationships"
        )

    return DynamicGraph





#----------------------HELPER FUNCTIONS----------------------#



class G_RAG:
    def __init__(
        self, 
        llm: BaseLanguageModel,
        allowed_nodes: List[str] = [], # CORE_NODE_TYPES by default?
        allowed_relationships: Union[List[str], List[Tuple[str, str, str]]] = [], # CORE_RELATIONSHIP_TYPES?
        prompt: Optional[ChatPromptTemplate] = None,
        node_properties: Union[bool, List[str]] = False,
        relationship_properties: Union[bool, List[str]] = False,
        ignore_tool_usage: bool = False,
        additional_instructions: str = "",


    ) -> None: 
        # init variables
        self._relationship_type = validate_and_get_relationship_type(
            allowed_relationships, allowed_nodes
        )
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self._function_call = ignore_tool_usage
        
        #init llm based on function calls
        if  self._function_call:
            prompt = prompt or create_unstructured_prompt(
                allowed_nodes or CORE_NODE_TYPES, # Pass core types if none specified
                allowed_relationships or CORE_RELATIONSHIP_TYPES, # Pass core types if none specified
                self._relationship_type,
                additional_instructions,
            )
            self.chain = prompt | llm
        else:
            try:
                llm_type = llm._llm_type  # type: ignore
            except AttributeError:
                llm_type = None
            schema = create_simple_model(
                allowed_nodes or CORE_NODE_TYPES, # Pass core types if none specified
                allowed_relationships or CORE_RELATIONSHIP_TYPES, # Pass core types if none specified
                node_properties,
                llm_type,
                relationship_properties,
                self._relationship_type,
            )
            structured_llm = llm.with_structured_output(schema, include_raw=True)
            prompt = prompt or get_default_prompt(additional_instructions)
            self.chain = prompt | structured_llm
            
            
        
        

        
        

