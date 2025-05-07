import asyncio
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, create_model

# --- NEW: Define Core Generic Types for Generalized Extraction ---
CORE_NODE_TYPES = ["Entity", "Concept", "Event", "Location", "Organization", "Person", "Time", "Document", "Property", "Condition", "Other"]
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

# Example of using tuples for more constraint (Optional, use if specific patterns are critical)
# CORE_RELATIONSHIP_TUPLES = [
#    ("Person", "WORKS_FOR", "Organization"),
#    ("Event", "OCCURRED_AT", "Location"),
#    ("Event", "OCCURRED_ON", "Time"),
#    ("Entity", "HAS_PROPERTY", "Property"),
#    ("Document", "MENTIONS", "Entity"),
# ]
# --- END NEW ---

DEFAULT_NODE_TYPE = "Entity" # Changed default to be more generic

# Keep existing examples or refine them to use CORE types/relations
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

# --- NEW: Updated System Prompt for Structured Output LLMs ---
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
# --- END NEW ---

def get_default_prompt(
    additional_instructions: str = "",
) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
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


def _get_additional_info(input_type: str) -> str:
    # Check if the input_type is one of the allowed values
    if input_type not in ["node", "relationship", "property"]:
        raise ValueError("input_type must be 'node', 'relationship', or 'property'")

    # Perform actions based on the input_type
    if input_type == "node":
        return (
            "Ensure you use basic or elementary types for node labels.\n"
            "For example, when you identify an entity representing a person, "
            "always label it as **'Person'**. Avoid using more specific terms "
            "like 'Mathematician' or 'Scientist'"
        )
    elif input_type == "relationship":
        return (
            "Instead of using specific and momentary types such as "
            "'BECAME_PROFESSOR', use more general and timeless relationship types "
            "like 'PROFESSOR'. However, do not sacrifice any accuracy for generality"
        )
    elif input_type == "property":
        return ""
    return ""

def optional_enum_field(
    enum_values: Optional[Union[List[str], List[Tuple[str, str, str]]]] = None,
    description: str = "",
    input_type: str = "node", # node, relationship, property
    llm_type: Optional[str] = None,
    relationship_type: Optional[str] = None,
    **field_kwargs: Any,
) -> Any:
    """Utility function to conditionally create a field with an enum constraint."""
    # --- CHANGED: Add guidance towards CORE types in description ---
    core_types_guidance = ""
    if not enum_values: # Only add general guidance if specific enums aren't provided
        if input_type == "node":
            core_types_guidance = f" Prefer general types like: {', '.join(CORE_NODE_TYPES)}."
        elif input_type == "relationship":
            core_types_guidance = f" Prefer general types like: {', '.join(CORE_RELATIONSHIP_TYPES)}."

    parsed_enum_values = enum_values
    # ... (rest of the enum handling logic remains the same) ...

    if enum_values and llm_type == "openai-chat":
        return Field(
            ...,
            enum=parsed_enum_values,  # type: ignore[call-arg]
            description=f"{description}. Available options: {parsed_enum_values}", # Keep specific options if provided
            **field_kwargs,
        ) # type: ignore[call-overload]
    elif enum_values:
        return Field(
            ...,
            description=f"{description}. Available options: {parsed_enum_values}", # Keep specific options if provided
            **field_kwargs,
        )
    else:
        # Add the general guidance if no specific enum values are given
        return Field(..., description=description + core_types_guidance + _get_additional_info(input_type), **field_kwargs)
class _Graph(BaseModel):
    nodes: Optional[List]
    relationships: Optional[List]


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


def create_unstructured_prompt(
    node_labels: Optional[List[str]] = None, # Suggest passing CORE_NODE_TYPES here
    rel_types: Optional[Union[List[str], List[Tuple[str, str, str]]]] = None, # Suggest passing CORE_RELATIONSHIP_TYPES here
    relationship_type: Optional[str] = None,
    additional_instructions: Optional[str] = "",
) -> ChatPromptTemplate:
    # --- CHANGED: Updated instructions to mention CORE types ---
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


def create_simple_model(
    node_labels: Optional[List[str]] = None,
    rel_types: Optional[Union[List[str], List[Tuple[str, str, str]]]] = None,
    node_properties: Union[bool, List[str]] = False,
    llm_type: Optional[str] = None,
    relationship_properties: Union[bool, List[str]] = False,
    relationship_type: Optional[str] = None,
) -> Type[_Graph]:
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

    node_fields: Dict[str, Tuple[Any, Any]] = {
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

    class DynamicGraph(_Graph):
        """Represents a graph document consisting of nodes and relationships."""

        nodes: Optional[List[SimpleNode]] = Field(description="List of nodes")  # type: ignore
        relationships: Optional[List[SimpleRelationship]] = Field(  # type: ignore
            description="List of relationships"
        )

    return DynamicGraph


def map_to_base_node(node: Any) -> Node:
    """Map the SimpleNode to the base Node."""
    properties = {}
    if hasattr(node, "properties") and node.properties:
        for p in node.properties:
            properties[format_property_key(p.key)] = p.value
    return Node(id=node.id, type=node.type, properties=properties)


def map_to_base_relationship(rel: Any) -> Relationship:
    """Map the SimpleRelationship to the base Relationship."""
    source = Node(id=rel.source_node_id, type=rel.source_node_type)
    target = Node(id=rel.target_node_id, type=rel.target_node_type)
    properties = {}
    if hasattr(rel, "properties") and rel.properties:
        for p in rel.properties:
            properties[format_property_key(p.key)] = p.value
    return Relationship(
        source=source, target=target, type=rel.type, properties=properties
    )


def _parse_and_clean_json(
    argument_json: Dict[str, Any],
) -> Tuple[List[Node], List[Relationship]]:
    nodes = []
    for node in argument_json["nodes"]:
        if not node.get("id"):  # Id is mandatory, skip this node
            continue
        node_properties = {}
        if "properties" in node and node["properties"]:
            for p in node["properties"]:
                node_properties[format_property_key(p["key"])] = p["value"]
        nodes.append(
            Node(
                id=node["id"],
                type=node.get("type", DEFAULT_NODE_TYPE),
                properties=node_properties,
            )
        )
    relationships = []
    for rel in argument_json["relationships"]:
        # Mandatory props
        if (
            not rel.get("source_node_id")
            or not rel.get("target_node_id")
            or not rel.get("type")
        ):
            continue

        # Node type copying if needed from node list
        if not rel.get("source_node_type"):
            try:
                rel["source_node_type"] = [
                    el.get("type")
                    for el in argument_json["nodes"]
                    if el["id"] == rel["source_node_id"]
                ][0]
            except IndexError:
                rel["source_node_type"] = DEFAULT_NODE_TYPE
        if not rel.get("target_node_type"):
            try:
                rel["target_node_type"] = [
                    el.get("type")
                    for el in argument_json["nodes"]
                    if el["id"] == rel["target_node_id"]
                ][0]
            except IndexError:
                rel["target_node_type"] = DEFAULT_NODE_TYPE

        rel_properties = {}
        if "properties" in rel and rel["properties"]:
            for p in rel["properties"]:
                rel_properties[format_property_key(p["key"])] = p["value"]

        source_node = Node(
            id=rel["source_node_id"],
            type=rel["source_node_type"],
        )
        target_node = Node(
            id=rel["target_node_id"],
            type=rel["target_node_type"],
        )
        relationships.append(
            Relationship(
                source=source_node,
                target=target_node,
                type=rel["type"],
                properties=rel_properties,
            )
        )
    return nodes, relationships


def _format_nodes(nodes: List[Node]) -> List[Node]:
    return [
        Node(
            id=el.id.title() if isinstance(el.id, str) else el.id,
            type=el.type.capitalize()  # type: ignore[arg-type]
            if el.type
            else DEFAULT_NODE_TYPE,  # handle empty strings  # type: ignore[arg-type]
            properties=el.properties,
        )
        for el in nodes
    ]


def _format_relationships(rels: List[Relationship]) -> List[Relationship]:
    return [
        Relationship(
            source=_format_nodes([el.source])[0],
            target=_format_nodes([el.target])[0],
            type=el.type.replace(" ", "_").upper(),
            properties=el.properties,
        )
        for el in rels
    ]


def format_property_key(s: str) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)


def _convert_to_graph_document(
    raw_schema: Dict[Any, Any],
) -> Tuple[List[Node], List[Relationship]]:
    # If there are validation errors
    if not raw_schema["parsed"]:
        try:
            try:  # OpenAI type response
                argument_json = json.loads(
                    raw_schema["raw"].additional_kwargs["tool_calls"][0]["function"][
                        "arguments"
                    ]
                )
            except Exception:  # Google type response
                try:
                    argument_json = json.loads(
                        raw_schema["raw"].additional_kwargs["function_call"][
                            "arguments"
                        ]
                    )
                except Exception:  # Ollama type response
                    argument_json = raw_schema["raw"].tool_calls[0]["args"]
                    if isinstance(argument_json["nodes"], str):
                        argument_json["nodes"] = json.loads(argument_json["nodes"])
                    if isinstance(argument_json["relationships"], str):
                        argument_json["relationships"] = json.loads(
                            argument_json["relationships"]
                        )
            nodes, relationships = _parse_and_clean_json(argument_json)
        except Exception:  # If we can't parse JSON
            return ([], [])
    else:  # If there are no validation errors use parsed pydantic object
        parsed_schema: _Graph = raw_schema["parsed"]
        nodes = (
            [map_to_base_node(node) for node in parsed_schema.nodes if node.id]
            if parsed_schema.nodes
            else []
        )

        relationships = (
            [
                map_to_base_relationship(rel)
                for rel in parsed_schema.relationships
                if rel.type and rel.source_node_id and rel.target_node_id
            ]
            if parsed_schema.relationships
            else []
        )
    # Title / Capitalize
    return _format_nodes(nodes), _format_relationships(relationships)


def validate_and_get_relationship_type(
    allowed_relationships: Union[List[str], List[Tuple[str, str, str]]],
    allowed_nodes: Optional[List[str]],
) -> Optional[str]:
    if allowed_relationships and not isinstance(allowed_relationships, list):
        raise ValueError("`allowed_relationships` attribute must be a list.")
    # If it's an empty list
    if not allowed_relationships:
        return None
    # Validate list of strings
    if all(isinstance(item, str) for item in allowed_relationships):
        # Valid: all items are strings, no further checks needed.
        return "string"

    # Validate list of 3-tuples and check if first/last elements are in allowed_nodes
    if all(
        isinstance(item, tuple)
        and len(item) == 3
        and all(isinstance(subitem, str) for subitem in item)
        and item[0] in allowed_nodes  # type: ignore
        and item[2] in allowed_nodes  # type: ignore
        for item in allowed_relationships
    ):
        # all items are 3-tuples, and the first/last elements are in allowed_nodes.
        return "tuple"

    # If the input doesn't match any of the valid cases, raise a ValueError
    raise ValueError(
        "`allowed_relationships` must be list of strings or a list of 3-item tuples. "
        "For tuples, the first and last elements must be in the `allowed_nodes` list."
    )

### Post processing 
# --- NEW: Post-processing Functions ---

def _merge_nodes_relationships(
    nodes: List[Node], relationships: List[Relationship]
) -> Tuple[List[Node], List[Relationship]]:
    """Merges duplicate nodes and relationships."""
    # Node merging: Based on ID and Type (case-insensitive for robustness)
    merged_nodes_dict: Dict[Tuple[str, str], Node] = {}
    for node in nodes:
        node_key = (str(node.id).lower(), str(node.type).lower())
        if node_key not in merged_nodes_dict:
            merged_nodes_dict[node_key] = node
        else:
            # Optional: Merge properties if needed, here we just keep the first one
            pass # Or implement property merging logic

    merged_nodes = list(merged_nodes_dict.values())
    node_map = { (str(n.id).lower(), str(n.type).lower()): n for n in merged_nodes } # Map lowercased id/type to actual merged node object

    # Relationship merging: Based on Source, Target, and Type (case-insensitive)
    merged_relationships_set = set()
    final_relationships = []

    for rel in relationships:
        # Find the corresponding *merged* source and target nodes
        source_key = (str(rel.source.id).lower(), str(rel.source.type).lower())
        target_key = (str(rel.target.id).lower(), str(rel.target.type).lower())

        merged_source_node = node_map.get(source_key)
        merged_target_node = node_map.get(target_key)

        # Only add relationship if both merged nodes exist
        if merged_source_node and merged_target_node:
            rel_key = (merged_source_node.id, merged_target_node.id, str(rel.type).upper()) # Use merged node IDs and uppercase type

            if rel_key not in merged_relationships_set:
                merged_relationships_set.add(rel_key)
                # Create new relationship with potentially merged nodes
                final_relationships.append(
                    Relationship(
                        source=merged_source_node,
                        target=merged_target_node,
                        type=str(rel.type).upper().replace(" ", "_"), # Ensure consistent formatting
                        properties=rel.properties # Keep properties from the first instance
                    )
                )

    return merged_nodes, final_relationships


# --- Placeholder Functions for Advanced Post-processing ---

def _apply_coreference_resolution(
    nodes: List[Node], relationships: List[Relationship], text: str
) -> Tuple[List[Node], List[Relationship]]:
    """
    Placeholder for coreference resolution.
    Requires integrating libraries like spaCy + neuralcoref or huggingface.
    Logic would involve:
    1. Running coreference resolution on the original 'text'.
    2. Identifying clusters of mentions referring to the same entity.
    3. Mapping nodes in the 'nodes' list to these clusters.
    4. Consolidating nodes/relationships based on the identified coreferences.
    """
    print("Warning: Coreference resolution not implemented. Skipping.")
    # This function should return the updated nodes and relationships lists
    # after applying coreference logic. For now, it returns inputs unchanged.
    return nodes, relationships


def _apply_entity_linking(nodes: List[Node]) -> List[Node]:
    """
    Placeholder for entity linking.
    Requires integrating entity linking libraries/models (e.g., spaCy entity linker, custom models).
    Logic would involve:
    1. Iterating through 'nodes' (especially Person, Organization, Location).
    2. Querying an entity linking service/model with the node ID (name) and context.
    3. Adding the canonical URI/ID (e.g., Wikidata ID) as a property to the node.
    """
    print("Warning: Entity linking not implemented. Skipping.")
    # This function should return the updated nodes list with linking information.
    # For now, it returns the input list unchanged.
    # Example modification:
    # for node in nodes:
    #    if node.type in ["Person", "Organization", "Location"]:
    #        linked_id = get_entity_link(node.id) # Fictional function
    #        if linked_id:
    #           node.properties["canonical_id"] = linked_id
    return nodes

def _apply_graph_validation(
    nodes: List[Node], relationships: List[Relationship]
) -> Tuple[List[Node], List[Relationship]]:
    """
    Placeholder for advanced graph validation rules.
    Examples:
    - Check relationship type plausibility between source/target node types.
    - Identify disconnected nodes or components.
    - Flag nodes with overly generic IDs.
    """
    print("Warning: Graph validation not implemented. Skipping.")
    # Could potentially filter nodes/relationships based on rules
    return nodes, relationships

# --- END NEW ---
class LLMGraphTransformer:
    # ... (keep existing __init__ parameters)
    def __init__(
        self,
        llm: BaseLanguageModel,
        allowed_nodes: List[str] = [], # Consider passing CORE_NODE_TYPES by default?
        allowed_relationships: Union[List[str], List[Tuple[str, str, str]]] = [], # Consider CORE_RELATIONSHIP_TYPES?
        prompt: Optional[ChatPromptTemplate] = None,
        strict_mode: bool = True,
        node_properties: Union[bool, List[str]] = False,
        relationship_properties: Union[bool, List[str]] = False,
        ignore_tool_usage: bool = False,
        additional_instructions: str = "",
        # --- NEW: Optional flags for post-processing ---
        apply_coreference: bool = False, # Default to False as it needs setup
        apply_entity_linking: bool = False, # Default to False
        apply_validation: bool = False, # Default to False
        # --- END NEW ---
    ) -> None:
        # ... (keep existing validation logic for relationships) ...
        self._relationship_type = validate_and_get_relationship_type(
            allowed_relationships, allowed_nodes
        )
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self.strict_mode = strict_mode
        self._function_call = not ignore_tool_usage
        # --- NEW: Store post-processing flags ---
        self._apply_coreference = apply_coreference
        self._apply_entity_linking = apply_entity_linking
        self._apply_validation = apply_validation
        # --- END NEW ---

        # ... (keep existing LLM capability check) ...

        if not self._function_call:
            # ... (keep non-function-call setup) ...
            # --- CHANGED: Default prompt uses CORE types if specific ones aren't given ---
            prompt = prompt or create_unstructured_prompt(
                allowed_nodes or CORE_NODE_TYPES, # Pass core types if none specified
                allowed_relationships or CORE_RELATIONSHIP_TYPES, # Pass core types if none specified
                self._relationship_type,
                additional_instructions,
            )
            self.chain = prompt | llm
            # --- END CHANGED ---
        else:
            # --- CHANGED: Default prompt uses system_prompt_v2 ---
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
            # Use the new V2 prompt by default
            prompt = prompt or get_default_prompt(additional_instructions)
            self.chain = prompt | structured_llm
            # --- END CHANGED ---

    def process_response(
        self, document: Document, config: Optional[RunnableConfig] = None
    ) -> GraphDocument:
        """
        Processes a single document, transforming it into a graph document using
        an LLM based on the model's schema and constraints. Includes post-processing.
        """
        text = document.page_content
        raw_schema = self.chain.invoke({"input": text}, config=config)

        nodes: List[Node] = []
        relationships: List[Relationship] = []

        # --- Initial Extraction Logic (Function Call vs Non-Function Call) ---
        if self._function_call:
            raw_schema = cast(Dict[Any, Any], raw_schema)
            nodes, relationships = _convert_to_graph_document(raw_schema)
        else:
            # --- Non-Function Call Parsing (Simplified from original) ---
            nodes_set = set()
            relationships_list = [] # Temporary list before formatting
            if not isinstance(raw_schema, str):
                raw_schema_content = getattr(raw_schema, 'content', str(raw_schema)) # Handle AIMessage etc.
            else:
                raw_schema_content = raw_schema

            try:
                parsed_json = self.json_repair.loads(raw_schema_content)
                if isinstance(parsed_json, dict): # Handle cases where LLM returns a single dict instead of list
                    parsed_json = [parsed_json]

                for rel in parsed_json:
                    if (
                        not isinstance(rel, dict)
                        or not rel.get("head")
                        or not rel.get("tail")
                        or not rel.get("relation")
                    ):
                        continue
                    head_type = rel.get("head_type", DEFAULT_NODE_TYPE)
                    tail_type = rel.get("tail_type", DEFAULT_NODE_TYPE)
                    nodes_set.add((rel["head"], head_type))
                    nodes_set.add((rel["tail"], tail_type))
                    relationships_list.append(
                        {
                            "source_id": rel["head"], "source_type": head_type,
                            "target_id": rel["tail"], "target_type": tail_type,
                            "type": rel["relation"], "properties": {} # Add properties if needed
                        }
                    )
            except Exception as e:
                print(f"Error parsing LLM (non-function call) output: {e}\nOutput:\n{raw_schema_content}")
                # Fallback to empty graph if parsing fails completely
                nodes, relationships = [], []


            # Create initial Node/Relationship objects from parsed data
            nodes = [Node(id=el[0], type=el[1]) for el in list(nodes_set)]
            temp_node_dict = { (n.id, n.type): n for n in nodes } # Quick lookup

            relationships = []
            for rel_data in relationships_list:
                source_node = temp_node_dict.get((rel_data["source_id"], rel_data["source_type"]))
                target_node = temp_node_dict.get((rel_data["target_id"], rel_data["target_type"]))
                if source_node and target_node:
                    relationships.append(Relationship(source=source_node, target=target_node, type=rel_data["type"], properties=rel_data["properties"]))
            # --- End Non-Function Call Parsing ---

        # --- CHANGED: Apply Post-processing Steps ---
        # 1. Merge duplicates (always recommended)
        nodes, relationships = _merge_nodes_relationships(nodes, relationships)

        # 2. Optional Coreference Resolution
        if self._apply_coreference:
            nodes, relationships = _apply_coreference_resolution(nodes, relationships, text)
            # Re-run merging after coref as IDs might have changed/consolidated
            nodes, relationships = _merge_nodes_relationships(nodes, relationships)

        # 3. Optional Entity Linking
        if self._apply_entity_linking:
            nodes = _apply_entity_linking(nodes) # Modifies nodes in-place or returns new list

        # 4. Optional Graph Validation
        if self._apply_validation:
            nodes, relationships = _apply_graph_validation(nodes, relationships)

        # --- END CHANGED ---


        # --- Strict mode filtering (Applied AFTER post-processing) ---
        if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
            # ... (Keep the existing strict mode filtering logic) ...
            if self.allowed_nodes:
                lower_allowed_nodes = [el.lower() for el in self.allowed_nodes]
                nodes = [
                    node for node in nodes if str(node.type).lower() in lower_allowed_nodes # Use str() for safety
                ]
                 # Filter relationships based on the *remaining* valid nodes
                valid_node_keys = { (str(n.id).lower(), str(n.type).lower()) for n in nodes }
                relationships = [
                    rel
                    for rel in relationships
                    if (str(rel.source.id).lower(), str(rel.source.type).lower()) in valid_node_keys
                    and (str(rel.target.id).lower(), str(rel.target.type).lower()) in valid_node_keys
                ]

            if self.allowed_relationships:
                # Filter by type and direction
                if self._relationship_type == "tuple":
                    allowed_rel_tuples_lower = { # Use a set for faster lookup
                        (str(s_t).lower(), str(r_t).lower(), str(t_t).lower())
                        for s_t, r_t, t_t in self.allowed_relationships
                    }
                    relationships = [
                        rel
                        for rel in relationships
                        if (
                            str(rel.source.type).lower(),
                            str(rel.type).lower(),
                            str(rel.target.type).lower(),
                        )
                        in allowed_rel_tuples_lower
                    ]
                else:  # Filter by type only
                    allowed_rel_types_lower = { str(el).lower() for el in self.allowed_relationships }
                    relationships = [
                        rel
                        for rel in relationships
                        if str(rel.type).lower() in allowed_rel_types_lower
                    ]


        return GraphDocument(nodes=nodes, relationships=relationships, source=document)

    # --- Update aprocess_response similarly ---
    async def aprocess_response(
        self, document: Document, config: Optional[RunnableConfig] = None
    ) -> GraphDocument:
        """
        Asynchronously processes a single document, transforming it into a
        graph document. Includes post-processing.
        """
        text = document.page_content
        raw_schema = await self.chain.ainvoke({"input": text}, config=config)

        nodes: List[Node] = []
        relationships: List[Relationship] = []

        # --- Initial Extraction Logic (Function Call vs Non-Function Call) ---
        # (Repeat the same extraction logic as in process_response)
        if self._function_call:
            raw_schema = cast(Dict[Any, Any], raw_schema)
            nodes, relationships = _convert_to_graph_document(raw_schema)
        else:
            # (Repeat the non-function call parsing logic from process_response)
            nodes_set = set()
            relationships_list = [] # Temporary list before formatting
            if not isinstance(raw_schema, str):
                raw_schema_content = getattr(raw_schema, 'content', str(raw_schema))
            else:
                raw_schema_content = raw_schema

            try:
                # NOTE: json_repair is sync, consider async json parsing if performance critical
                parsed_json = self.json_repair.loads(raw_schema_content)
                if isinstance(parsed_json, dict):
                    parsed_json = [parsed_json]

                for rel in parsed_json:
                    if (
                        not isinstance(rel, dict)
                        or not rel.get("head")
                        or not rel.get("tail")
                        or not rel.get("relation")
                    ):
                        continue
                    head_type = rel.get("head_type", DEFAULT_NODE_TYPE)
                    tail_type = rel.get("tail_type", DEFAULT_NODE_TYPE)
                    nodes_set.add((rel["head"], head_type))
                    nodes_set.add((rel["tail"], tail_type))
                    relationships_list.append(
                        {
                            "source_id": rel["head"], "source_type": head_type,
                            "target_id": rel["tail"], "target_type": tail_type,
                            "type": rel["relation"], "properties": {}
                        }
                    )
            except Exception as e:
                print(f"Error parsing LLM (non-function call) output [async]: {e}\nOutput:\n{raw_schema_content}")
                nodes, relationships = [], []

            nodes = [Node(id=el[0], type=el[1]) for el in list(nodes_set)]
            temp_node_dict = { (n.id, n.type): n for n in nodes }

            relationships = []
            for rel_data in relationships_list:
                source_node = temp_node_dict.get((rel_data["source_id"], rel_data["source_type"]))
                target_node = temp_node_dict.get((rel_data["target_id"], rel_data["target_type"]))
                if source_node and target_node:
                    relationships.append(Relationship(source=source_node, target=target_node, type=rel_data["type"], properties=rel_data["properties"]))


        # --- Apply Post-processing Steps (same as sync version) ---
        nodes, relationships = _merge_nodes_relationships(nodes, relationships)

        if self._apply_coreference:
            # Note: _apply_coreference_resolution is currently sync,
            # make it async or run in executor if it becomes I/O bound
            nodes, relationships = _apply_coreference_resolution(nodes, relationships, text)
            nodes, relationships = _merge_nodes_relationships(nodes, relationships)

        if self._apply_entity_linking:
            # Note: _apply_entity_linking is currently sync
            nodes = _apply_entity_linking(nodes)

        if self._apply_validation:
            # Note: _apply_graph_validation is currently sync
            nodes, relationships = _apply_graph_validation(nodes, relationships)


        # --- Strict mode filtering (same as sync version) ---
        if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
            # ... (Repeat the existing strict mode filtering logic) ...
            if self.allowed_nodes:
                lower_allowed_nodes = [el.lower() for el in self.allowed_nodes]
                nodes = [
                    node for node in nodes if str(node.type).lower() in lower_allowed_nodes
                ]
                valid_node_keys = { (str(n.id).lower(), str(n.type).lower()) for n in nodes }
                relationships = [
                    rel
                    for rel in relationships
                    if (str(rel.source.id).lower(), str(rel.source.type).lower()) in valid_node_keys
                    and (str(rel.target.id).lower(), str(rel.target.type).lower()) in valid_node_keys
                ]

            if self.allowed_relationships:
                if self._relationship_type == "tuple":
                    allowed_rel_tuples_lower = {
                        (str(s_t).lower(), str(r_t).lower(), str(t_t).lower())
                        for s_t, r_t, t_t in self.allowed_relationships
                    }
                    relationships = [
                        rel
                        for rel in relationships
                        if (
                            str(rel.source.type).lower(),
                            str(rel.type).lower(),
                            str(rel.target.type).lower(),
                        )
                        in allowed_rel_tuples_lower
                    ]
                else:
                    allowed_rel_types_lower = { str(el).lower() for el in self.allowed_relationships }
                    relationships = [
                        rel
                        for rel in relationships
                        if str(rel.type).lower() in allowed_rel_types_lower
                    ]

        return GraphDocument(nodes=nodes, relationships=relationships, source=document)

    # ... (convert_to_graph_documents and aconvert_to_graph_documents remain the same) ...
    
    def convert_to_graph_documents(
        self, documents: Sequence[Document], config: Optional[RunnableConfig] = None
    ) -> List[GraphDocument]:
        """Convert a sequence of documents into graph documents.

        Args:
            documents (Sequence[Document]): The original documents.
            kwargs: Additional keyword arguments.

        Returns:
            Sequence[GraphDocument]: The transformed documents as graphs.
        """
        return [self.process_response(document, config) for document in documents]

    
    async def aconvert_to_graph_documents(
        self, documents: Sequence[Document], config: Optional[RunnableConfig] = None
    ) -> List[GraphDocument]:
        """
        Asynchronously convert a sequence of documents into graph documents.
        """
        tasks = [
            asyncio.create_task(self.aprocess_response(document, config))
            for document in documents
        ]
        results = await asyncio.gather(*tasks)
        return results


# --- END OF CLASS ---class LLMGraphTransformer:
