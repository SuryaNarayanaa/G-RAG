import asyncio
import hashlib
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


from sentence_transformers import SentenceTransformer


DEFAULT_NODE_TYPE = "Node"

examples = [
    {
        "text": (
            "Adam is a software engineer in Microsoft since 2009, "
            "and last year he got an award as the Best Talent"
        ),
        "head": "Adam",
        "head_type": "Person",
        "relation": "WORKS_FOR",
        "tail": "Microsoft",
        "tail_type": "Company",
    },
    {
        "text": (
            "Adam is a software engineer in Microsoft since 2009, "
            "and last year he got an award as the Best Talent"
        ),
        "head": "Adam",
        "head_type": "Person",
        "relation": "HAS_AWARD",
        "tail": "Best Talent",
        "tail_type": "Award",
    },
    {
        "text": (
            "Microsoft is a tech company that provide "
            "several products such as Microsoft Word"
        ),
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "PRODUCED_BY",
        "tail": "Microsoft",
        "tail_type": "Company",
    },
    {
        "text": "Microsoft Word is a lightweight app that accessible offline",
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "HAS_CHARACTERISTIC",
        "tail": "lightweight app",
        "tail_type": "Characteristic",
    },
    {
        "text": "Microsoft Word is a lightweight app that accessible offline",
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "HAS_CHARACTERISTIC",
        "tail": "accessible offline",
        "tail_type": "Characteristic",
    },
]

system_prompt = (
    "# Knowledge Graph Instructions for GPT-4\n"
    "## 1. Overview\n"
    "You are a top-tier algorithm designed for extracting information in structured "
    "formats to build a knowledge graph.\n"
    "Try to capture as much information from the text as possible without "
    "sacrificing accuracy. Do not add any information that is not explicitly "
    "mentioned in the text.\n"
    "- **Nodes** represent entities and concepts.\n"
    "- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n"
    "accessible for a vast audience.\n"
    "## 2. Labeling Nodes\n"
    "- **Consistency**: Ensure you use available types for node labels.\n"
    "Ensure you use basic or elementary types for node labels.\n"
    "- For example, when you identify an entity representing a person, "
    "always label it as **'person'**. Avoid using more specific terms "
    "like 'mathematician' or 'scientist'."
    "- **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
    "names or human-readable identifiers found in the text.\n"
    "- **Relationships** represent connections between entities or concepts.\n"
    "Ensure consistency and generality in relationship types when constructing "
    "knowledge graphs. Instead of using specific and momentary types "
    "such as 'BECAME_PROFESSOR', use more general and timeless relationship types "
    "like 'PROFESSOR'. Make sure to use general and timeless relationship types!\n"
    "## 3. Coreference Resolution\n"
    "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
    "ensure consistency.\n"
    'If an entity, such as "John Doe", is mentioned multiple times in the text '
    'but is referred to by different names or pronouns (e.g., "Joe", "he"),'
    "always use the most complete identifier for that entity throughout the "
    'knowledge graph. In this example, use "John Doe" as the entity ID.\n'
    "Remember, the knowledge graph should be coherent and easily understandable, "
    "so maintaining consistency in entity references is crucial.\n"
    "## 4. Strict Compliance\n"
    "Adhere to the rules strictly. Non-compliance will result in termination."
)


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
    input_type: str = "node",
    llm_type: Optional[str] = None,
    relationship_type: Optional[str] = None,
    **field_kwargs: Any,
) -> Any:
    parsed_enum_values = enum_values
    # We have to extract enum types from tuples
    if relationship_type == "tuple":
        parsed_enum_values = list({el[1] for el in enum_values})  # type: ignore

    # Only openai supports enum param
    if enum_values and llm_type == "openai-chat":
        return Field(
            ...,
            enum=parsed_enum_values,  # type: ignore[call-arg]
            description=f"{description}. Available options are {parsed_enum_values}",
            **field_kwargs,
        )
    elif enum_values:
        return Field(
            ...,
            description=f"{description}. Available options are {parsed_enum_values}",
            **field_kwargs,
        )
    else:
        additional_info = _get_additional_info(input_type)
        return Field(..., description=description + additional_info, **field_kwargs)


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
    node_labels: Optional[List[str]] = None,
    rel_types: Optional[Union[List[str], List[Tuple[str, str, str]]]] = None,
    relationship_type: Optional[str] = None,
    additional_instructions: Optional[str] = "",
) -> ChatPromptTemplate:
    node_labels_str = str(node_labels) if node_labels else ""
    if rel_types:
        if relationship_type == "tuple":
            rel_types_str = str(list({item[1] for item in rel_types}))
        else:
            rel_types_str = str(rel_types)
    else:
        rel_types_str = ""
    base_string_parts = [
        "You are a top-tier algorithm designed for extracting information in "
        "structured formats to build a knowledge graph. Your task is to identify "
        "the entities and relations requested with the user prompt from a given "
        "text. You must generate the output in a JSON format containing a list "
        'with JSON objects. Each object should have the keys: "head", '
        '"head_type", "relation", "tail", and "tail_type". The "head" '
        "key must contain the text of the extracted entity with one of the types "
        "from the provided list in the user prompt.",
        f'The "head_type" key must contain the type of the extracted head entity, '
        f"which must be one of the types from {node_labels_str}."
        if node_labels
        else "",
        f'The "relation" key must contain the type of relation between the "head" '
        f'and the "tail", which must be one of the relations from {rel_types_str}.'
        if rel_types
        else "",
        f'The "tail" key must represent the text of an extracted entity which is '
        f'the tail of the relation, and the "tail_type" key must contain the type '
        f"of the tail entity from {node_labels_str}."
        if node_labels
        else "",
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
        "that entity. The knowledge graph should be coherent and easily "
        "understandable, so maintaining consistency in entity references is "
        "crucial.",
        "IMPORTANT NOTES:\n- Don't add any explanation and text. ",
        additional_instructions,
    ]
    system_prompt = "\n".join(filter(None, base_string_parts))

    system_message = SystemMessage(content=system_prompt)
    parser = JsonOutputParser(pydantic_object=UnstructuredRelation)

    human_string_parts = [
        "Based on the following example, extract entities and "
        "relations from the provided text.\n\n",
        "Use the following entity types, don't use other entity "
        "that is not defined below:"
        "# ENTITY TYPES:"
        "{node_labels}"
        if node_labels
        else "",
        "Use the following relation types, don't use other relation "
        "that is not defined below:"
        "# RELATION TYPES:"
        "{rel_types}"
        if rel_types
        else "",
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
            "node_labels": node_labels,
            "rel_types": rel_types,
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
        nodes: Optional[List[SimpleNode]] = Field(description="List of nodes")  # type: ignore
        relationships: Optional[List[SimpleRelationship]] = Field(  # type: ignore
            description="List of relationships"
        )

    return DynamicGraph


def map_to_base_node(node: Any) -> Node:
    properties = {}
    if hasattr(node, "properties") and node.properties:
        for p in node.properties:
            properties[format_property_key(p.key)] = p.value
    return Node(id=node.id, type=node.type, properties=properties)


def map_to_base_relationship(rel: Any) -> Relationship:
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
        if (
            not rel.get("source_node_id")
            or not rel.get("target_node_id")
            or not rel.get("type")
        ):
            continue

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


class LLMGraphTransformer:
    def __init__(
        self,
        llm: BaseLanguageModel,
        allowed_nodes: List[str] = [],
        allowed_relationships: Union[List[str], List[Tuple[str, str, str]]] = [],
        prompt: Optional[ChatPromptTemplate] = None,
        strict_mode: bool = True,
        node_properties: Union[bool, List[str]] = False,
        relationship_properties: Union[bool, List[str]] = False,
        ignore_tool_usage: bool = False,
        additional_instructions: str = "",
        embed_model_name: str = "all-mpnet-base-v2",
        
    ) -> None:
        print("Init LLMGraphTransformer")
        # Validate and check allowed relationships input
        self._relationship_type = validate_and_get_relationship_type(
            allowed_relationships, allowed_nodes
        )
        self.embedder = SentenceTransformer(embed_model_name)
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self.strict_mode = strict_mode
        self._function_call = not ignore_tool_usage

        # Check if the LLM really supports structured output
        if self._function_call:
            try:
                llm.with_structured_output(_Graph)
            except NotImplementedError:
                self._function_call = False
        if not self._function_call:
            if node_properties or relationship_properties:
                raise ValueError(
                    "The 'node_properties' and 'relationship_properties' parameters "
                    "cannot be used in combination with a LLM that doesn't support "
                    "native function calling."
                )
            try:
                import json_repair  # type: ignore

                self.json_repair = json_repair
            except ImportError:
                raise ImportError(
                    "Could not import json_repair python package. "
                    "Please install it with `pip install json-repair`."
                )
            prompt = prompt or create_unstructured_prompt(
                allowed_nodes,
                allowed_relationships,
                self._relationship_type,
                additional_instructions,
            )
            self.chain = prompt | llm
        else:
            # Define chain
            try:
                llm_type = llm._llm_type  # type: ignore
            except AttributeError:
                llm_type = None
            schema = create_simple_model(
                allowed_nodes,
                allowed_relationships,
                node_properties,
                llm_type,
                relationship_properties,
                self._relationship_type,
            )
            structured_llm = llm.with_structured_output(schema, include_raw=True)
            prompt = prompt or get_default_prompt(additional_instructions)
            self.chain = prompt | structured_llm
            
    def _embed_text(self, text: str) -> List[float]:
        vec = self.embedder.encode(text, normalize_embeddings=True, show_progress_bar=True ,)
        return vec.tolist()
    def process_response(
        self, document: Document, config: Optional[RunnableConfig] = None
    ) -> GraphDocument:
        try:
            text = document.page_content
            raw_schema = self.chain.invoke({"input": text}, config=config)
            
            # Block 1: Function call handling
            if self._function_call:
                try:
                    raw_schema = cast(Dict[Any, Any], raw_schema)
                    nodes, relationships = _convert_to_graph_document(raw_schema)
                except (TypeError, ValueError) as e:
                    print(f"Error in function call processing: {e}")
                    nodes, relationships = [], []
            else:
                try:
                    nodes_set = set()
                    relationships = []
                    if not isinstance(raw_schema, str):
                        raw_schema = raw_schema.content
                    parsed_json = self.json_repair.loads(raw_schema)
                    if isinstance(parsed_json, dict):
                        parsed_json = [parsed_json]
                        
                    # Block 2: Relationship and node processing
                    for rel in parsed_json:
                        try:
                            if (
                                not isinstance(rel, dict)
                                or not rel.get("head")
                                or not rel.get("tail")
                                or not rel.get("relation")
                            ):
                                continue
                                
                            head = rel["head"] if isinstance(rel["head"], (str, int)) else str(rel["head"])
                            tail = rel["tail"] if isinstance(rel["tail"], (str, int)) else str(rel["tail"])

                            nodes_set.add((head, rel.get("head_type", DEFAULT_NODE_TYPE)))
                            nodes_set.add((tail, rel.get("tail_type", DEFAULT_NODE_TYPE)))

                            source_node = Node(
                                id=head, type=rel.get("head_type", DEFAULT_NODE_TYPE)
                            )
                            target_node = Node(
                                id=tail, type=rel.get("tail_type", DEFAULT_NODE_TYPE)
                            )
                            relationships.append(
                                Relationship(
                                    source=source_node, target=target_node, type=rel["relation"]
                                )
                            )
                        except (TypeError, ValueError) as e:
                            print(f"Error processing relationship: {e}")
                            continue
                            
                    nodes = [Node(id=el[0], type=el[1]) for el in list(nodes_set)]
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    print(f"Error in JSON processing: {e}")
                    nodes, relationships = [], []

            # Block 3: Passage node creation
            try:
                passage_id = hashlib.md5(text.encode()).hexdigest()
                passage_node = Node(
                    id=passage_id,
                    type="Passage",
                    properties={
                        "text": text,
                        "source_doc_id": document.metadata.get("doc_id"),
                        "source_page": document.metadata.get("page"),
                    }
                )
                passage_node.properties["embedding"] = self._embed_text(text)
            except (TypeError, ValueError) as e:
                print(f"Error creating passage node: {e}")
                passage_node = Node(id="error_passage", type="Passage")

            # Block 4: MENTIONS relationships
            try:
                mentions_rels = [
                    Relationship(
                        source=passage_node,
                        target=Node(
                            id=n.id, 
                            type=n.type,
                            properties=n.properties.copy() if n.properties else {}
                        ),
                        type="MENTIONS",
                        properties={"source_text": text}
                    )
                    for n in nodes
                ]
            except (TypeError, ValueError) as e:
                print(f"Error creating MENTIONS relationships: {e}")
                mentions_rels = []

            # Block 5: Node properties
            for node in nodes:
                try:
                    node.properties.setdefault("source_texts", [])
                    node.properties["source_texts"].append(text)
                    if hasattr(document, "metadata"):
                        if document.metadata.get("doc_id"):
                            node.properties["source_doc_id"] = document.metadata["doc_id"]
                        if document.metadata.get("page"):
                            node.properties["source_page"] = document.metadata["page"]
                except (TypeError, AttributeError) as e:
                    print(f"Error setting node properties: {e}")
                    continue

            # Block 6: Relationship properties
            for rel in relationships:
                try:
                    rel.properties["source_text"] = text
                except (TypeError, AttributeError) as e:
                    print(f"Error setting relationship properties: {e}")
                    continue

            # Block 7: Strict mode filtering
            if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
                try:
                    if self.allowed_nodes:
                        lower_allowed_nodes = [el.lower() for el in self.allowed_nodes]
                        nodes = [
                            node for node in nodes if node.type.lower() in lower_allowed_nodes
                        ]
                        relationships = [
                            rel
                            for rel in relationships
                            if rel.source.type.lower() in lower_allowed_nodes
                            and rel.target.type.lower() in lower_allowed_nodes
                        ]
                    if self.allowed_relationships:
                        if self._relationship_type == "tuple":
                            relationships = [
                                rel
                                for rel in relationships
                                if (
                                    (
                                        rel.source.type.lower(),
                                        rel.type.lower(),
                                        rel.target.type.lower(),
                                    )
                                    in [
                                        (s_t.lower(), r_t.lower(), t_t.lower())
                                        for s_t, r_t, t_t in self.allowed_relationships
                                    ]
                                )
                            ]
                        else:
                            relationships = [
                                rel
                                for rel in relationships
                                if rel.type.lower()
                                in [el.lower() for el in self.allowed_relationships]
                            ]
                except (TypeError, AttributeError) as e:
                    print(f"Error in strict mode filtering: {e}")

            return GraphDocument(
                nodes=[passage_node] + nodes,
                relationships=mentions_rels + relationships,
                source=document
            )
            
        except Exception as e:
            print(f"Critical error in process_response: {e}")
            return GraphDocument(
                nodes=[Node(id="error", type="Error")],
                relationships=[],
                source=document
            )

    def convert_to_graph_documents(
        self, documents: Sequence[Document], config: Optional[RunnableConfig] = None
    ) -> List[GraphDocument]:
        try:
            print("In -- convert_to_graph_documents")
            chunk_graphs: List[GraphDocument] = [
                self.process_response(doc, config) for doc in documents
            ]

            # Block 8: Node merging
            print("Merging Nodes")
            merged_nodes: dict[str, Node] = {}
            try:
                for gd in chunk_graphs:
                    for node in gd.nodes:
                        try:
                            snippets = node.properties.get("source_texts") or []
                            if node.id in merged_nodes:
                                merged_nodes[node.id].properties["source_texts"].extend(snippets)
                            else:
                                new_node = Node(
                                    id=node.id,
                                    type=node.type,
                                    properties={ **node.properties, "source_texts": list(snippets) }
                                )
                                merged_nodes[node.id] = new_node
                        except (TypeError, AttributeError) as e:
                            print(f"Error merging node: {e}")
                            continue
            except Exception as e:
                print(f"Error in node merging: {e}")
                merged_nodes = {}

            # Block 9: Relationship merging
            print("Merging relationships")
            merged_rels: List[Relationship] = []
            try:
                for gd in chunk_graphs:
                    merged_rels.extend(gd.relationships)
            except (TypeError, AttributeError) as e:
                print(f"Error merging relationships: {e}")

            # Block 10: Computing embeddings
            print("Computing embeddings for nodes")
            for node in merged_nodes.values():
                try:
                    context = " ".join(node.properties.get("source_texts", []))
                    node.properties["embedding"] = self._embed_text(context)
                except (TypeError, ValueError) as e:
                    print(f"Error computing node embedding: {e}")

            print("Computing embeddings for relationships")
            for rel in merged_rels:
                try:
                    text = rel.properties.get("source_text", "")
                    if text:
                        rel.properties["embedding"] = self._embed_text(text)
                except (TypeError, ValueError) as e:
                    print(f"Error computing relationship embedding: {e}")

            print("Building final GraphDocument")
            return [GraphDocument(
                nodes=list(merged_nodes.values()),
                relationships=merged_rels,
                source=documents[0]
            )]
            
        except Exception as e:
            print(f"Critical error in convert_to_graph_documents: {e}")
            return [GraphDocument(
                nodes=[Node(id="error", type="Error")],
                relationships=[],
                source=documents[0]
            )]

    async def aprocess_response(
        self, document: Document, config: Optional[RunnableConfig] = None
    ) -> GraphDocument:
        text = document.page_content
        raw_schema = await self.chain.ainvoke({"input": text}, config=config)
        if self._function_call:
            raw_schema = cast(Dict[Any, Any], raw_schema)
            nodes, relationships = _convert_to_graph_document(raw_schema)
        else:
            nodes_set = set()
            relationships = []
            if not isinstance(raw_schema, str):
                raw_schema = raw_schema.content
            parsed_json = self.json_repair.loads(raw_schema)
            if isinstance(parsed_json, dict):
                parsed_json = [parsed_json]
            for rel in parsed_json:
                # Check if mandatory properties are there
                if (
                    not rel.get("head")
                    or not rel.get("tail")
                    or not rel.get("relation")
                ):
                    continue
                # Nodes need to be deduplicated using a set
                # Use default Node label for nodes if missing
                nodes_set.add((rel["head"], rel.get("head_type", DEFAULT_NODE_TYPE)))
                nodes_set.add((rel["tail"], rel.get("tail_type", DEFAULT_NODE_TYPE)))

                source_node = Node(
                    id=rel["head"], type=rel.get("head_type", DEFAULT_NODE_TYPE)
                )
                target_node = Node(
                    id=rel["tail"], type=rel.get("tail_type", DEFAULT_NODE_TYPE)
                )
                relationships.append(
                    Relationship(
                        source=source_node, target=target_node, type=rel["relation"]
                    )
                )
            # Create nodes list
            nodes = [Node(id=el[0], type=el[1]) for el in list(nodes_set)]

        if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
            if self.allowed_nodes:
                lower_allowed_nodes = [el.lower() for el in self.allowed_nodes]
                nodes = [
                    node for node in nodes if node.type.lower() in lower_allowed_nodes
                ]
                relationships = [
                    rel
                    for rel in relationships
                    if rel.source.type.lower() in lower_allowed_nodes
                    and rel.target.type.lower() in lower_allowed_nodes
                ]
            if self.allowed_relationships:
                # Filter by type and direction
                if self._relationship_type == "tuple":
                    relationships = [
                        rel
                        for rel in relationships
                        if (
                            (
                                rel.source.type.lower(),
                                rel.type.lower(),
                                rel.target.type.lower(),
                            )
                            in [  # type: ignore
                                (s_t.lower(), r_t.lower(), t_t.lower())
                                for s_t, r_t, t_t in self.allowed_relationships
                            ]
                        )
                    ]
                else:  # Filter by type only
                    relationships = [
                        rel
                        for rel in relationships
                        if rel.type.lower()
                        in [el.lower() for el in self.allowed_relationships]  # type: ignore
                    ]

        return GraphDocument(nodes=nodes, relationships=relationships, source=document)

    async def aconvert_to_graph_documents(
        self, documents: Sequence[Document], config: Optional[RunnableConfig] = None
    ) -> List[GraphDocument]:
        tasks = [
            asyncio.create_task(self.aprocess_response(document, config))
            for document in documents
        ]
        results = await asyncio.gather(*tasks)
        return results
