from lc_neo4j import Neo4jGraph

def setup_graph_with_embeddings(graph_documents, neo4j_graph: Neo4jGraph):
    # 1. First add the documents to Neo4j
    neo4j_graph.add_graph_documents(
        graph_documents=graph_documents,
        include_source=True,  # This will include the original document text
        baseEntityLabel=False  # Set to True if you want a base Entity label
    )

    # 2. Get all unique node types and relationship types from the documents
    node_types = set()
    relationship_types = set()
    
    for doc in graph_documents:
        # Collect all node types
        for node in doc.nodes:
            if hasattr(node, 'properties') and 'embedding' in node.properties:
                node_types.add(node.type)
        
        # Collect all relationship types
        for rel in doc.relationships:
            if hasattr(rel, 'properties') and 'embedding' in rel.properties:
                relationship_types.add(rel.type)

    # 3. Create vector indexes for each node type that has embeddings
    for node_type in node_types:
        index_name = f"{node_type.lower()}Embeddings"
        neo4j_graph.create_vector_index(
            index_name=index_name,
            label=node_type,
            property_name="embedding",
            dimension=neo4j_graph.embedding_dim
        )

    # 4. Create vector indexes for each relationship type that has embeddings
    for rel_type in relationship_types:
        index_name = f"{rel_type.lower()}Embeddings"
        neo4j_graph.create_vector_index(
            index_name=index_name,
            label=rel_type,
            property_name="embedding",
            dimension=neo4j_graph.embedding_dim,
            is_relationship=True
        )

def get_vector_index_name(type_name: str) -> str:
    """Generate consistent index name for a node or relationship type."""
    return f"{type_name.lower()}Embeddings"

def vector_search(neo4j_graph: Neo4jGraph, query_embedding: list, node_type: str, 
                 k: int = 5, is_relationship: bool = False) -> list:
    """
    Perform vector similarity search for either nodes or relationships.
    
    Args:
        neo4j_graph: Neo4jGraph instance
        query_embedding: The query vector to search with
        node_type: The type of node or relationship to search
        k: Number of results to return
        is_relationship: Whether to search relationships instead of nodes
        
    Returns:
        List of tuples containing (node/relationship properties, score)
    """
    index_name = get_vector_index_name(node_type)
    
    if is_relationship:
        cypher = """
        CALL db.index.vector.queryRelationships($index_name, $k, $embedding)
        YIELD relationship, score
        RETURN relationship AS item, score
        ORDER BY score DESC
        """
    else:
        cypher = """
        CALL db.index.vector.queryNodes($index_name, $k, $embedding)
        YIELD node AS item, score
        RETURN item, score
        ORDER BY score DESC
        """
    
    results = neo4j_graph.query(
        cypher,
        params={
            'index_name': index_name,
            'k': k,
            'embedding': query_embedding
        }
    )
    
    return [(record["item"], record["score"]) for record in results]

def search_similar_nodes(neo4j_graph: Neo4jGraph, query_embedding: list, 
                        node_types: list = None, k: int = 5) -> dict:
    """
    Search for similar nodes across multiple node types.
    
    Args:
        neo4j_graph: Neo4jGraph instance
        query_embedding: The query vector to search with
        node_types: List of node types to search. If None, will search all indexed types.
        k: Number of results to return per node type
        
    Returns:
        Dict mapping node types to their search results
    """
    # If no node types specified, get all node types that have vector indexes
    if node_types is None:
        # This cypher query gets all node labels that have vector indexes
        cypher = """
        CALL db.indexes()
        YIELD name, type, labelsOrTypes
        WHERE type = 'VECTOR'
        AND name ENDS WITH 'Embeddings'
        RETURN distinct labelsOrTypes[0] as nodeType
        """
        results = neo4j_graph.query(cypher)
        node_types = [r["nodeType"] for r in results]
    
    results = {}
    for node_type in node_types:
        try:
            type_results = vector_search(
                neo4j_graph=neo4j_graph,
                query_embedding=query_embedding,
                node_type=node_type,
                k=k
            )
            results[node_type] = type_results
        except Exception as e:
            print(f"Warning: Could not search node type {node_type}: {str(e)}")
            
    return results

def search_similar_relationships(neo4j_graph: Neo4jGraph, query_embedding: list, 
                               relationship_types: list = None, k: int = 5) -> dict:
    """
    Search for similar relationships across multiple relationship types.
    
    Args:
        neo4j_graph: Neo4jGraph instance
        query_embedding: The query vector to search with
        relationship_types: List of relationship types to search. If None, will search all indexed types.
        k: Number of results to return per relationship type
        
    Returns:
        Dict mapping relationship types to their search results
    """
    # If no relationship types specified, get all relationship types that have vector indexes
    if relationship_types is None:
        cypher = """
        CALL db.indexes()
        YIELD name, type, labelsOrTypes
        WHERE type = 'VECTOR'
        AND name ENDS WITH 'Embeddings'
        RETURN distinct labelsOrTypes[0] as relType
        """
        results = neo4j_graph.query(cypher)
        relationship_types = [r["relType"] for r in results]
    
    results = {}
    for rel_type in relationship_types:
        try:
            type_results = vector_search(
                neo4j_graph=neo4j_graph,
                query_embedding=query_embedding,
                node_type=rel_type,
                k=k,
                is_relationship=True
            )
            results[rel_type] = type_results
        except Exception as e:
            print(f"Warning: Could not search relationship type {rel_type}: {str(e)}")
            
    return results

# Usage example:
'''
# Initialize Neo4j connection
neo4j_graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="your_password",
    embedding_dim=768  # Set this to match your embedding dimension
)

# After getting graph_documents from LLMGraphTransformer
setup_graph_with_embeddings(graph_documents, neo4j_graph)

# Create an embedding for your query text using the same model as in LLMGraphTransformer
query_text = "What is machine learning?"
query_embedding = llm_transformer.embedder.encode(query_text, normalize_embeddings=True).tolist()

# Search across all node types
results = search_similar_nodes(neo4j_graph, query_embedding)

# Or search specific node types
person_results = search_similar_nodes(
    neo4j_graph, 
    query_embedding,
    node_types=['Person', 'Organization']
)

# Search relationships
relationship_results = search_similar_relationships(
    neo4j_graph, 
    query_embedding,
    relationship_types=['MENTIONS', 'WORKS_FOR']
)

# Print results
for node_type, matches in results.items():
    print(f"\nResults for {node_type}:")
    for node, score in matches:
        print(f"Score: {score:.4f}")
        print(f"Properties: {node}")
'''
