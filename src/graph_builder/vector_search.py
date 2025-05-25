from typing import List, Dict, Tuple, Any, Optional
from lc_neo4j import Neo4jGraph

def get_vector_index_name(type_name: str) -> str:
    """Generate consistent index name for a node or relationship type."""
    return f"{type_name.lower()}Embeddings"

def vector_search(
    neo4j_graph: Neo4jGraph,
    query_embedding: List[float],
    node_type: str,
    k: int = 5,
    is_relationship: bool = False
) -> List[Tuple[Dict[str, Any], float]]:
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

def search_similar_nodes(
    neo4j_graph: Neo4jGraph,
    query_embedding: List[float],
    node_types: Optional[List[str]] = None,
    k: int = 5
) -> Dict[str, List[Tuple[Dict[str, Any], float]]]:
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

def search_similar_relationships(
    neo4j_graph: Neo4jGraph,
    query_embedding: List[float],
    relationship_types: Optional[List[str]] = None,
    k: int = 5
) -> Dict[str, List[Tuple[Dict[str, Any], float]]]:
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
