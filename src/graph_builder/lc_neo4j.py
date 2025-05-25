from hashlib import md5
from neo4j import GraphDatabase
from typing import Any, Dict, List, Optional, Type

import neo4j
from langchain_core.utils import get_from_dict_or_env
from neo4j_graphrag.schema import (
    BASE_ENTITY_LABEL,
    _value_sanitize,
    format_schema,
    get_structured_schema,
)

from langchain_neo4j.graphs.graph_document import GraphDocument
from langchain_neo4j.graphs.graph_store import GraphStore

include_docs_query = (
    "MERGE (d:Document {id:$document.metadata.id}) "
    "SET d.text = $document.page_content "
    "SET d += $document.metadata "
    "WITH d "
)


def _get_node_import_query(baseEntityLabel: bool, include_source: bool) -> str:
    if baseEntityLabel:
        return (
            f"{include_docs_query if include_source else ''}"
            "UNWIND $data AS row "
            f"MERGE (source:`{BASE_ENTITY_LABEL}` {{id: row.id}}) "
            "SET source += row.properties "
            f"{'MERGE (d)-[:MENTIONS]->(source) ' if include_source else ''}"
            "WITH source, row "
            "CALL apoc.create.addLabels( source, [row.type] ) YIELD node "
            "RETURN distinct 'done' AS result"
        )
    else:
        return (
            f"{include_docs_query if include_source else ''}"
            "UNWIND $data AS row "
            "CALL apoc.merge.node([row.type], {id: row.id}, "
            "row.properties, {}) YIELD node "
            f"{'MERGE (d)-[:MENTIONS]->(node) ' if include_source else ''}"
            "RETURN distinct 'done' AS result"
        )


def _get_rel_import_query(baseEntityLabel: bool) -> str:
    if baseEntityLabel:
        return (
            "UNWIND $data AS row "
            f"MERGE (source:`{BASE_ENTITY_LABEL}` {{id: row.source}}) "
            f"MERGE (target:`{BASE_ENTITY_LABEL}` {{id: row.target}}) "
            "WITH source, target, row "
            "CALL apoc.merge.relationship(source, row.type, "
            "{}, row.properties, target) YIELD rel "
            "RETURN distinct 'done'"
        )
    else:
        return (
            "UNWIND $data AS row "
            "CALL apoc.merge.node([row.source_label], {id: row.source},"
            "{}, {}) YIELD node as source "
            "CALL apoc.merge.node([row.target_label], {id: row.target},"
            "{}, {}) YIELD node as target "
            "CALL apoc.merge.relationship(source, row.type, "
            "{}, row.properties, target) YIELD rel "
            "RETURN distinct 'done'"
        )


def _remove_backticks(text: str) -> str:
    return text.replace("`", "")


class Neo4jGraph(GraphStore):
    """Neo4j database wrapper for various graph operations.

    Parameters:
    url (Optional[str]): The URL of the Neo4j database server.
    username (Optional[str]): The username for database authentication.
    password (Optional[str]): The password for database authentication.
    database (str): The name of the database to connect to. Default is 'neo4j'.
    timeout (Optional[float]): The timeout for transactions in seconds.
            Useful for terminating long-running queries.
            By default, there is no timeout set.
    sanitize (bool): A flag to indicate whether to remove lists with
            more than 128 elements from results. Useful for removing
            embedding-like properties from database responses. Default is False.
    refresh_schema (bool): A flag whether to refresh schema information
            at initialization. Default is True.
    enhanced_schema (bool): A flag whether to scan the database for
            example values and use them in the graph schema. Default is False.
    driver_config (Dict): Configuration passed to Neo4j Driver.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        timeout: Optional[float] = None,
        sanitize: bool = False,
        refresh_schema: bool = True,
        *,
        driver_config: Optional[Dict] = None,
        enhanced_schema: bool = False,
                embedding_dim: int,
    ) -> None:
        """Create a new Neo4j graph wrapper instance."""
        print("INIT Neo4jGraph -5")
        url = get_from_dict_or_env({"url": url}, "url", "NEO4J_URI")
        # if username and password are "", assume Neo4j auth is disabled
        if username == "" and password == "":
            auth = None
        else:
            username = get_from_dict_or_env(
                {"username": username},
                "username",
                "NEO4J_USERNAME",
            )
            password = get_from_dict_or_env(
                {"password": password},
                "password",
                "NEO4J_PASSWORD",
            )
            auth = (username, password)
        database = get_from_dict_or_env(
            {"database": database}, "database", "NEO4J_DATABASE", "neo4j"
        )

        self._driver = neo4j.GraphDatabase.driver(
            url, auth=auth, **(driver_config or {})
        )
        self._database = database
        self.timeout = timeout
        self.sanitize = sanitize
        self._enhanced_schema = enhanced_schema
        self.schema: str = ""
        self.structured_schema: Dict[str, Any] = {}
        self.embedding_dim =embedding_dim
        # Verify connection
        try:
            self._driver.verify_connectivity()
        except neo4j.exceptions.ConfigurationError:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the driver config is correct"
            )
        except neo4j.exceptions.ServiceUnavailable:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the url is correct"
            )
        except neo4j.exceptions.AuthError:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the username and password are correct"
            )
        # Set schema
        if refresh_schema:
            try:
                self.refresh_schema()
            except neo4j.exceptions.ClientError as e:
                if e.code == "Neo.ClientError.Procedure.ProcedureNotFound":
                    raise ValueError(
                        "Could not use APOC procedures. "
                        "Please ensure the APOC plugin is installed in Neo4j and that "
                        "'apoc.meta.data()' is allowed in Neo4j configuration "
                    )
                raise e

    def _check_driver_state(self) -> None:
        """
        Check if the driver is available and ready for operations.

        Raises:
            RuntimeError: If the driver has been closed or is not initialized.
        """
        if not hasattr(self, "_driver"):
            raise RuntimeError(
                "Cannot perform operations - Neo4j connection has been closed"
            )

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Graph"""
        return self.schema

    @property
    def get_structured_schema(self) -> Dict[str, Any]:
        """Returns the structured schema of the Graph"""
        return self.structured_schema

    def query(
        self,
        query: str,
        params: dict = {},
        session_params: dict = {},
    ) -> List[Dict[str, Any]]:
        """Query Neo4j database.

        Args:
            query (str): The Cypher query to execute.
            params (dict): The parameters to pass to the query.
            session_params (dict): Parameters to pass to the session used for executing
                the query.

        Returns:
            List[Dict[str, Any]]: The list of dictionaries containing the query results.

        Raises:
            RuntimeError: If the connection has been closed.
        """
        self._check_driver_state()
        from neo4j import Query
        from neo4j.exceptions import Neo4jError

        if not session_params:
            try:
                data, _, _ = self._driver.execute_query(
                    Query(text=query, timeout=self.timeout),
                    database_=self._database,
                    parameters_=params,
                )
                json_data = [r.data() for r in data]
                if self.sanitize:
                    json_data = [_value_sanitize(el) for el in json_data]
                return json_data
            except Neo4jError as e:
                if not (
                    (
                        (  # isCallInTransactionError
                            e.code == "Neo.DatabaseError.Statement.ExecutionFailed"
                            or e.code
                            == "Neo.DatabaseError.Transaction.TransactionStartFailed"
                        )
                        and e.message is not None
                        and "in an implicit transaction" in e.message
                    )
                    or (  # isPeriodicCommitError
                        e.code == "Neo.ClientError.Statement.SemanticError"
                        and e.message is not None
                        and (
                            "in an open transaction is not possible" in e.message
                            or "tried to execute in an explicit transaction"
                            in e.message
                        )
                    )
                ):
                    raise
        # fallback to allow implicit transactions
        session_params.setdefault("database", self._database)
        with self._driver.session(**session_params) as session:
            result = session.run(Query(text=query, timeout=self.timeout), params)
            json_data = [r.data() for r in result]
            if self.sanitize:
                json_data = [_value_sanitize(el) for el in json_data]
            return json_data

    def refresh_schema(self) -> None:
        """
        Refreshes the Neo4j graph schema information.

        Raises:
            RuntimeError: If the connection has been closed.
        """
        self._check_driver_state()
        self.structured_schema = get_structured_schema(
            driver=self._driver,
            is_enhanced=self._enhanced_schema,
            database=self._database,
            timeout=self.timeout,
            sanitize=self.sanitize,
        )
        self.schema = format_schema(
            schema=self.structured_schema, is_enhanced=self._enhanced_schema
        )
        
    def retrieve_passages(self, query_vector: List[float], top_k: int = 5):
        cypher = """
        WITH $q AS qvec
        CALL db.index.vector.queryNodes('passageEmbeddings', $k, qvec)
        YIELD node AS p, score
        RETURN p.id AS passageId, p.text AS snippet, score
        ORDER BY score DESC
        """
        return self.query(cypher, params={'q':query_vector, 'k':top_k})

    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
        baseEntityLabel: bool = False,
    ) -> None:
        self._check_driver_state()

        # 1) Ensure __Entity__ uniqueness constraint
        if baseEntityLabel:
            constraint_exists = any(
                el["labelsOrTypes"] == [BASE_ENTITY_LABEL]
                and el["properties"] == ["id"]
                for el in self.structured_schema
                .get("metadata", {})
                .get("constraint", [])
            )
            if not constraint_exists:
                # create the constraint
                self.query(
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (b:{BASE_ENTITY_LABEL}) REQUIRE b.id IS UNIQUE"
                )
                self.refresh_schema()

        # **Removed Index Creation Logic**

        # 3) Optional: enforce that every document has metadata if include_source
        if include_source:
            for doc in graph_documents:
                if doc.source is None:
                    raise TypeError(
                        "include_source=True but at least one GraphDocument has no source"
                    )

        # 4) Build import queries
        node_import_query = _get_node_import_query(baseEntityLabel, include_source)
        rel_import_query = _get_rel_import_query(baseEntityLabel)

        for document in graph_documents:
            # 4a) Prepare node import parameters
            node_params: Dict[str, Any] = {
                "data": [n.__dict__ for n in document.nodes]
            }
            if include_source and document.source:
                if not document.source.metadata.get("id"):
                    document.source.metadata["id"] = md5(
                        document.source.page_content.encode("utf-8")
                    ).hexdigest()
                node_params["document"] = document.source.__dict__

            # strip backticks from labels
            for n in document.nodes:
                n.type = _remove_backticks(n.type)  # Ensure valid labels

            # execute node import
            self.query(node_import_query, node_params)

            # 4b) Prepare and execute relationship import
            rel_data = []
            for r in document.relationships:
                rel_data.append({
                    "source":       r.source.id,
                    "source_label": _remove_backticks(r.source.type),
                    "target":       r.target.id,
                    "target_label": _remove_backticks(r.target.type),
                    "type":         _remove_backticks(r.type.replace(" ", "_").upper()),
                    "properties":   r.properties,
                })
            self.query(rel_import_query, {"data": rel_data})
    def create_vector_index(
        self,
        index_name: str,
        label: str,
        property_name: str,
        dimension: int,
        is_relationship: bool = False,
        relationship_type: str = None,
        similarity_function: str = "COSINE"
    ) -> None:
        """
        Create a vector index in Neo4j for nodes or relationships.

        Args:
            index_name: Name of the index.
            label: Node label or relationship type.
            property_name: Name of the property containing the embedding.
            dimension: Dimension of the embedding vector.
            is_relationship: If True, attempts to create a relationship index.
            relationship_type: Relationship type (required if is_relationship is True).
            similarity_function: The similarity function to use (either "COSINE" or "EUCLIDEAN").
        
        Raises:
            ValueError: If similarity_function is invalid or relationship index is requested
            Neo4jError: If index creation fails for any other reason than index already exists
        """
        self._check_driver_state()
        
        if is_relationship:
            # Relationship vector indexes are not supported in Neo4j
            print(f"Warning: Relationship vector indexes are not supported in Neo4j. "
                  f"The index creation for relationship type '{relationship_type}' will be skipped. "
                  f"Consider storing vector data in nodes instead.")
            return
            
        # Validate similarity function
        similarity_function = similarity_function.upper()
        if similarity_function not in ["COSINE", "EUCLIDEAN"]:
            raise ValueError("similarity_function must be either 'COSINE' or 'EUCLIDEAN'")
            
        # Use the modern vector index syntax
        cypher = (
            f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
            f"FOR (n:{label}) "
            f"ON (n.{property_name}) "
            f"OPTIONS {{ "
            f"indexConfig: {{ "
            f"`vector.dimensions`: {dimension}, "
            f"`vector.similarity_function`: '{similarity_function}' "
            f"}} "
            f"}}"
        )
        
        # Execute the query with better error handling
        try:
            self.query(cypher)
            print(
                f"Vector index '{index_name}' created for node label '{label}' "
                f"on property '{property_name}' with dimension {dimension} "
                f"using {similarity_function} similarity."
            )
        except Exception as e:
            error_msg = str(e)
            if "AlreadyIndexedException" in error_msg:
                print(f"Info: Index '{index_name}' already exists for {label}.{property_name}")
            elif "SyntaxError" in error_msg:
                raise ValueError(f"Invalid syntax in index creation. This may indicate "
                               f"that your Neo4j version doesn't support the modern vector "
                               f"index syntax. Error: {error_msg}")
            else:
                raise

    def _get_vector_index_name(self, type_name: str) -> str:
        """Generate consistent index name for a node or relationship type."""
        return f"{type_name.lower()}Embeddings"

    def vector_search(
        self,
        query_embedding: List[float],
        type_name: str,
        k: int = 5,
        is_relationship: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search for either nodes or relationships.
        
        Args:
            query_embedding: The query vector to search with
            type_name: The type of node or relationship to search
            k: Number of results to return
            is_relationship: Whether this is a relationship type
            
        Returns:
            List of dictionaries containing the matched items and their scores
            
        Note:
            Relationship vector search is not supported in Neo4j. The method will
            return an empty list for relationship searches.
        """
        self._check_driver_state()
        
        if is_relationship:
            print(f"Warning: Relationship vector search is not supported in Neo4j. "
                  f"Consider storing vector data in nodes instead. "
                  f"Skipping search for relationship type '{type_name}'.")
            return []
        
        # Use modern vector search syntax for nodes
        cypher = f"""
        MATCH (item:{type_name})
        WHERE item.embedding IS NOT NULL
        WITH item, vector.similarity.cosine(item.embedding, $embedding) AS score
        ORDER BY score DESC
        LIMIT $k
        RETURN item, score
        """
        
        try:
            return self.query(
                cypher,
                params={
                    'k': k,
                    'embedding': query_embedding
                }
            )
        except Exception as e:
            error_msg = str(e)
            if "no such vector schema index" in error_msg.lower():
                print(f"Warning: No vector index found for node type '{type_name}'. "
                      f"The index might not exist or the type name could be incorrect.")
                return []
            elif "SyntaxError" in error_msg:
                # Try falling back to old syntax if modern syntax fails
                # This should not normally happen as we create indexes with new syntax
                print("Warning: Modern vector search syntax not supported. "
                      "Trying legacy procedure...")
                cypher = """
                CALL db.index.vector.queryNodes($index_name, $k, $embedding)
                YIELD node AS item, score
                RETURN item, score
                ORDER BY score DESC
                """
                return self.query(
                    cypher,
                    params={
                        'index_name': self._get_vector_indexed_types(type_name),
                        'k': k,
                        'embedding': query_embedding
                    }
                )
            else:
                raise    
    def search_similar_nodes(
        self,
        query_embedding: List[float],
        node_types: Optional[List[str]] = None,
        k: int = 5,
        print_results: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for similar nodes across multiple node types.
        
        Args:
            query_embedding: The query vector to search with
            node_types: List of node types to search. If None, will search all indexed types.
            k: Number of results to return per node type
            print_results: Whether to print the results
            
        Returns:
            Dict mapping node types to their search results
        """
        self._check_driver_state()
        
        # If no node types specified, get all node types that have vector indexes
        if node_types is None:
            cypher = """
            SHOW INDEXES
            YIELD name, type, labelsOrTypes
            WHERE type = 'VECTOR'
            AND name ENDS WITH 'Embeddings'
            AND (labelsOrTypes IS NOT NULL AND size(labelsOrTypes) > 0)
            RETURN distinct labelsOrTypes[0] as nodeType
            """
            results = self.query(cypher)
            node_types = [r["nodeType"] for r in results]
        
        results = {}
        for node_type in node_types:
            try:
                type_results = self.vector_search(
                    query_embedding=query_embedding,
                    type_name=node_type,
                    k=k,
                    is_relationship=False
                )
                results[node_type] = type_results
            except Exception as e:
                print(f"Warning: Could not search node type {node_type}: {str(e)}")
        
        if print_results:
            self._print_top_results(results, k)
                
        return results

    def search_similar_relationships(
        self,
        query_embedding: List[float],
        relationship_types: Optional[List[str]] = None,
        k: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for similar relationships across multiple relationship types.
        
        Args:
            query_embedding: The query vector to search with
            relationship_types: List of relationship types to search. If None, will search all indexed types.
            k: Number of results to return per relationship type
            
        Returns:
            Dict mapping relationship types to their search results
        """
        self._check_driver_state()
        
        # If no relationship types specified, get all relationship types that have vector indexes
        if relationship_types is None:
            cypher = """
            SHOW INDEXES
            YIELD name, type, labelsOrTypes
            WHERE type = 'VECTOR'
            AND name ENDS WITH 'Embeddings'
            AND (labelsOrTypes IS NOT NULL AND size(labelsOrTypes) > 0)
            RETURN distinct labelsOrTypes[0] as relType
            """
            results = self.query(cypher)
            relationship_types = [r["relType"] for r in results]
        
        results = {}
        for rel_type in relationship_types:
            try:
                type_results = self.vector_search(
                    query_embedding=query_embedding,
                    type_name=rel_type,
                    k=k,
                    is_relationship=True
                )
                results[rel_type] = type_results
            except Exception as e:
                print(f"Warning: Could not search relationship type {rel_type}: {str(e)}")
                
        return results


    def create_vector_indexes_for_documents(
        self,
        graph_documents: List[GraphDocument],
    ) -> None:
        """
        Create vector indexes for all node and relationship types that have embeddings.
        
        Args:
            graph_documents: List of GraphDocument objects to analyze and create indexes for
        """
        # Get all unique node types and relationship types from the documents
        node_types = set()
        relationship_types = set()
        
        for doc in graph_documents:
            # Collect all node types that have embeddings
            for node in doc.nodes:
                if hasattr(node, 'properties') and 'embedding' in node.properties:
                    node_types.add(node.type)
            
            # Collect all relationship types that have embeddings
            for rel in doc.relationships:
                if hasattr(rel, 'properties') and 'embedding' in rel.properties:
                    relationship_types.add(rel.type)

        # Create vector indexes for each node type that has embeddings
        for node_type in node_types:
            index_name = self._get_vector_index_name(node_type)
            try:
                self.create_vector_index(
                    index_name=index_name,
                    label=node_type,
                    property_name="embedding",
                    dimension=self.embedding_dim
                )
            except Exception as e:
                print(f"Warning: Could not create index for node type {node_type}: {str(e)}")

        # Create vector indexes for each relationship type that has embeddings
        for rel_type in relationship_types:
            index_name = self._get_vector_index_name(rel_type)
            try:
                self.create_vector_index(
                    index_name=index_name,
                    label=rel_type,
                    property_name="embedding",
                    dimension=self.embedding_dim,
                    is_relationship=True
                )
            except Exception as e:
                print(f"Warning: Could not create index for relationship type {rel_type}: {str(e)}")
                
    def setup_graph_with_documents(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = True,
        base_entity_label: bool = False
    ) -> None:
        """
        Complete setup of the graph with documents - stores documents and creates all necessary indexes.
        
        Args:
            graph_documents: List of GraphDocument objects to store and index
            include_source: Whether to include source document information
            base_entity_label: Whether to use a base entity label for all nodes
        """
        # 1. First store the documents in Neo4j
        self.add_graph_documents(
            graph_documents=graph_documents,
            include_source=include_source,
            baseEntityLabel=base_entity_label
        )
        
        # 2. Create vector indexes for all types that have embeddings
        self.create_vector_indexes_for_documents(graph_documents)
    def close(self) -> None:
        """
        Explicitly close the Neo4j driver connection.

        Delegates connection management to the Neo4j driver.
        """
        if hasattr(self, "_driver"):
            self._driver.close()
            # Remove the driver attribute to indicate closure
            delattr(self, "_driver")

    def __enter__(self) -> "Neo4jGraph":
        """
        Enter the runtime context for the Neo4j graph connection.

        Enables use of the graph connection with the 'with' statement.
        This method allows for automatic resource management and ensures
        that the connection is properly handled.

        Returns:
            Neo4jGraph: The current graph connection instance

        Example:
            with Neo4jGraph(...) as graph:
                graph.query(...)  # Connection automatically managed
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """
        Exit the runtime context for the Neo4j graph connection.

        This method is automatically called when exiting a 'with' statement.
        It ensures that the database connection is closed, regardless of
        whether an exception occurred during the context's execution.

        Args:
            exc_type: The type of exception that caused the context to exit
                      (None if no exception occurred)
            exc_val: The exception instance that caused the context to exit
                     (None if no exception occurred)
            exc_tb: The traceback for the exception (None if no exception occurred)

        Note:
            Any exception is re-raised after the connection is closed.
        """
        self.close()

    def __del__(self) -> None:
        """
        Destructor for the Neo4j graph connection.

        This method is called during garbage collection to ensure that
        database resources are released if not explicitly closed.

        Caution:
            - Do not rely on this method for deterministic resource cleanup
            - Always prefer explicit .close() or context manager

        Best practices:
            1. Use context manager:
               with Neo4jGraph(...) as graph:
                   ...
            2. Explicitly close:
               graph = Neo4jGraph(...)
               try:
                   ...
               finally:
                   graph.close()
        """
        try:
            self.close()
        except Exception:
            # Suppress any exceptions during garbage collection
            pass

    def _get_vector_indexed_types(self) -> List[str]:
        """Get all types (nodes/relationships) that have vector indexes."""
        cypher = """
        SHOW INDEXES
        YIELD name, type, labelsOrTypes
        WHERE type = 'VECTOR'
        AND name ENDS WITH 'Embeddings'
        AND (labelsOrTypes IS NOT NULL AND size(labelsOrTypes) > 0)
        RETURN distinct labelsOrTypes[0] as type
        """
        results = self.query(cypher)
        return [r["type"] for r in results]

    def _print_top_results(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        k: int = 5,
        include_score: bool = True
    ) -> None:
        """
        Helper method to print the top k results from vector search.
        
        Args:
            results: Dictionary mapping types to their search results
            k: Number of top results to print per type
            include_score: Whether to include similarity scores in output
        """
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
                
                # Print result with optional score
                if include_score:
                    print(f"  {i+1}. [{result['score']:.3f}] {display_value}")
                else:
                    print(f"  {i+1}. {display_value}")

    def search_similar_nodes(
        self,
        query_embedding: List[float],
        node_types: Optional[List[str]] = None,
        k: int = 5,
        print_results: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for similar nodes across multiple node types.
        
        Args:
            query_embedding: The query vector to search with
            node_types: List of node types to search. If None, will search all indexed types.
            k: Number of results to return per node type
            print_results: Whether to print the results
            
        Returns:
            Dict mapping node types to their search results
        """
        self._check_driver_state()
        
        # If no node types specified, get all node types that have vector indexes
        if node_types is None:
            cypher = """
            SHOW INDEXES
            YIELD name, type, labelsOrTypes
            WHERE type = 'VECTOR'
            AND name ENDS WITH 'Embeddings'
            AND (labelsOrTypes IS NOT NULL AND size(labelsOrTypes) > 0)
            RETURN distinct labelsOrTypes[0] as nodeType
            """
            results = self.query(cypher)
            node_types = [r["nodeType"] for r in results]
        
        results = {}
        for node_type in node_types:
            try:
                type_results = self.vector_search(
                    query_embedding=query_embedding,
                    type_name=node_type,
                    k=k,
                    is_relationship=False
                )
                results[node_type] = type_results
            except Exception as e:
                print(f"Warning: Could not search node type {node_type}: {str(e)}")
        
        if print_results:
            self._print_top_results(results, k)
                
        return results

    def search_similar_relationships(
        self,
        query_embedding: List[float],
        relationship_types: Optional[List[str]] = None,
        k: int = 5,
        print_results: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for similar relationships across multiple relationship types.
        Note: Relationship vector search is not currently supported in Neo4j.
        
        Args:
            query_embedding: The query vector to search with
            relationship_types: List of relationship types to search. If None, will search all indexed types.
            k: Number of results to return per relationship type
            print_results: Whether to print the results
            
        Returns:
            Dict mapping relationship types to their search results (empty in current version)
        """
        self._check_driver_state()
        
        # If no relationship types specified, get all relationship types that have vector indexes
        if relationship_types is None:
            cypher = """
            SHOW INDEXES
            YIELD name, type, labelsOrTypes
            WHERE type = 'VECTOR'
            AND name ENDS WITH 'Embeddings'
            AND (labelsOrTypes IS NOT NULL AND size(labelsOrTypes) > 0)
            RETURN distinct labelsOrTypes[0] as relType
            """
            results = self.query(cypher)
            relationship_types = [r["relType"] for r in results]
        
        results = {}
        for rel_type in relationship_types:
            try:
                type_results = self.vector_search(
                    query_embedding=query_embedding,
                    type_name=rel_type,
                    k=k,
                    is_relationship=True
                )
                results[rel_type] = type_results
            except Exception as e:
                print(f"Warning: Could not search relationship type {rel_type}: {str(e)}")
        
        if print_results:
            self._print_top_results(results, k)
                
        return results
