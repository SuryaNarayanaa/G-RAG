import os
from ingestion.connectors.pdf_parser import PDFParser
from  ingestion.connectors.audio_parser import AudioParser
from  ingestion.connectors.image_parser import ImageParser
from  ingestion.connectors.video_parser import VideoParser

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import textwrap

from neo4j import GraphDatabase

class TripleExtractor:
    def __init__(self, model_name="tiiuae/falcon-7b-instruct", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device    = device

    def extract_triples(self, text_chunk: str) -> list[tuple]:
        """
        Given a text chunk, returns a list of (head, relation, tail) triples.
        """
        prompt = textwrap.dedent(f"""
        Extract all factual triples as JSON array.  
        Each triple should be ["Entity1", "Relation", "Entity2"].  
        Text:
        \"\"\"{text_chunk}\"\"\"
        """).strip()

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Assuming the model emits valid JSON array of triples
        try:
            triples = torch.json.loads(decoded)
            return [tuple(t) for t in triples]
        except Exception:
            # Fallback: parse lines like Entity1 | Relation | Entity2
            triples = []
            for line in decoded.splitlines():
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 3:
                    triples.append(tuple(parts))
            return triples# src/graph_builder/graph_builder.py

class GraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def upsert_triples(self, triples: list[tuple], source_meta: dict):
        """
        Inserts each (h, r, t) into the graph, attaching source metadata
        (e.g. file, page, offset) to each relationship.
        """
        with self.driver.session() as sess:
            for head, rel, tail in triples:
                sess.run(
                    """
                    MERGE (h:Entity {name: $head})
                    MERGE (t:Entity {name: $tail})
                    MERGE (h)-[r:REL {type: $rel}]-()
                    ON CREATE SET r += $meta
                    """,
                    head=head, tail=tail, rel=rel, meta=source_meta
                )


def process_pdf(pdf_path, graph, extractor):
    parser = PDFParser(pdf_path, output_dir="output/pdf/")
    text_file, _ = parser.extract_text_and_images()
    # Read and chunk text
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]

    for idx, chunk in enumerate(chunks, 1):
        triples = extractor.extract_triples(chunk)
        meta = {"source": pdf_path, "type": "pdf", "chunk": idx}
        graph.upsert_triples(triples, meta)

# Repeat similarly for images, audio, video...
# For audio:
def process_audio(audio_path, graph, extractor):
    audio_parser = AudioParser(audio_path)
    wav = f"output/audio/{os.path.basename(audio_path)}.wav"
    AudioParser.convert_to_wav(audio_path, wav)
    transcript = audio_parser.extract_text_with_whisper(output_dir="output/audio/")
    with open(transcript, 'r', encoding='utf-8') as f:
        text = f.read()
    triples = extractor.extract_triples(text)
    graph.upsert_triples(triples, {"source": audio_path, "type": "audio"})

def main():    # 1. Initialize
    extractor = TripleExtractor(model_name="tiiuae/falcon-7b-instruct")
    graph     = GraphBuilder("neo4j+s://ffa05957.databases.neo4j.io", "neo4j", "KYDXnv3miiiEFXu9p1119eo77ugFg_gcBqFxLlVD1h4")

    # 2. Process a PDF
    process_pdf("testing/1.pdf", graph, extractor)

    # 3. Process an Audio
    process_audio("testing/audio.wav", graph, extractor)

    # 4. Process an Image (text + OCR)
    image_parser = ImageParser("testing/image.png")
    img_data     = image_parser.extract_text_and_metadata()
    triples      = extractor.extract_triples(img_data["text"])
    graph.upsert_triples(triples, {"source": img_data["metadata"]["filename"], "type": "image"})

    # 5. Process a Video
    video_path = VideoParser.download_youtube_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    audio_path = VideoParser.extract_audio_from_video(video_path)
    transcript = AudioParser(audio_path).extract_text_with_whisper("output/video/")
    with open(transcript, 'r') as f: text = f.read()
    triples = extractor.extract_triples(text)
    graph.upsert_triples(triples, {"source": video_path, "type": "video"})

    graph.close()
