from src.integrated_pipeline import IntegratedGraphRAG

# Initialize the system
rag = IntegratedGraphRAG(
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="iambatman",
    # google_api_key="AIzaSyDM_XKj4CmHuX3reWEdT3cNVTxUIvhtwyg",  # Optional if set in environment
    output_dir="output"
)

# Process different types of files
# rag.process_document("D:\\Learning\\projects\\G-RAG\\testing\\tables2.pdf")
# rag.process_document("D:\\Learning\\projects\\G-RAG\\testing\\preprocessed_image.png")
# rag.process_document("path/to/video.mp4")
# rag.process_document("path/to/image.jpg")

# Query the system
response = rag.query("""ongqi An, Xu Zhao, Tao Yu, Ming Tang, and Jinqiao Wang.
Fluctuation-based Adaptive Structured Pruning for Large Language
Models.
[2] Kyoungtaek Choi, Seong Min Wi, Ho Gi Jung, and Jae Kyu Suhr.
Simplification of Deep Neural Network-Based Object Detector for Real-
Time Edge Computing.
[3] Viviana Crescitelli, Seiji Miura, Goichi Ono, and Naohiro Kohmu. Edge
Devices Object Detection by Filter Pruning.
[4] Giacomo Di Fabrizio, Lorenzo Calisti, Chiara Contoli, Nicholas Kania,
and Emanuele Lattanzi. A Study on the Energy-Efficiency of the Object
Tracking Algorithms in Edge Devices""")
print(response)

# Clean up when done
rag.close()