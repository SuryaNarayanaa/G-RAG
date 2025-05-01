import fitz 
import camelot
import os 
from pdf2docx import Converter
import csv
import pymupdf4llm
import pathlib

TESTING_DIR = 'testing/'
PDF_DIR = TESTING_DIR+'tables2.pdf'
OUTPUT_DIR = 'output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TESTING_DIR, exist_ok=True)



# -- table extraction 
cv = Converter(PDF_DIR)
tables = cv.extract_tables()
for idx, table in enumerate(tables, 1):
    csv_file = os.path.join(OUTPUT_DIR, f'table_{idx}.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(table)
cv.close()

## -- save as md for llm
md_text = pymupdf4llm.to_markdown("testing/1.pdf")
pathlib.Path("output.md").write_bytes(md_text.encode())

# -- text and image
with fitz.open(PDF_DIR) as doc:
    for page_index ,page in enumerate(doc, start=1):
        images = page.get_images()
        text = page.get_text()
        with open(OUTPUT_DIR + 'op.txt', 'a', encoding='utf-8') as f:

            f.write(text)
        for image_index, img in enumerate(images, start=1):
            xref = img[0] # get the XREF of the image
            pix = fitz.Pixmap(doc, xref) # create a Pixmap

            if pix.n - pix.alpha > 3: # CMYK: convert to RGB first
                pix = fitz.Pixmap(fitz.csRGB, pix)

            pix.save(OUTPUT_DIR + "page_%s-image_%s.png" % (page_index, image_index)) # save the image as png
            pix = None