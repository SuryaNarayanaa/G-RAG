import os
import csv
import fitz  # PyMuPDF
from pdf2docx import Converter
import pathlib
import pymupdf4llm


class PDFParser:
    def __init__(self, pdf_path, output_dir="output/"):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_tables(self):
        """Extract tables from PDF and save them as CSV files."""
        cv = Converter(self.pdf_path)
        tables = cv.extract_tables()
        table_paths = []

        for idx, table in enumerate(tables, 1):
            csv_file = os.path.join(self.output_dir, f'table_{idx}.csv')
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(table)
            table_paths.append(csv_file)

        cv.close()
        return table_paths

    def extract_text_and_images(self):
        """Extract text and images from PDF and save them to the output directory."""
        text_file_path = os.path.join(self.output_dir, 'op.txt')
        image_file_paths = []

        with fitz.open(self.pdf_path) as doc:
            for page_index, page in enumerate(doc, start=1):
                
                # Extract text
                text = page.get_text()
                with open(text_file_path, 'a', encoding='utf-8') as f:
                    f.write(text)

                # Extract images
                images = page.get_images(full=True)
                for image_index, img in enumerate(images, start=1):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)

                    if pix.n - pix.alpha > 3:  # Convert CMYK to RGB if necessary
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    image_path = os.path.join(self.output_dir, f"page_{page_index}-image_{image_index}.png")
                    pix.save(image_path)
                    image_file_paths.append(image_path)
                    pix = None  # Free up resources

        return text_file_path, image_file_paths

    def convert_to_markdown(self):
        """Convert PDF content to markdown format for LLM consumption."""
        markdown_text = pymupdf4llm.to_markdown(self.pdf_path)
        markdown_file_path = os.path.join(self.output_dir, "output.md")
        pathlib.Path(markdown_file_path).write_bytes(markdown_text.encode())
        return markdown_file_path
