import json
import os
from typing import Dict, Optional, Union

import cv2
import numpy as np
import pytesseract
from PIL import Image


class ImageParser:
    def __init__(self, image_path: str, output_dir: str = "output/"):
        self.image_path = image_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)  
        
    def extract_text_and_metadata(self) -> Optional[Dict[str, Union[str, dict]]]:
        """Extract text, pictures, tables and metadata from image for RAG processing."""
        try:
            image = cv2.imread(self.image_path)
            original_image = Image.open(self.image_path)
            
            # Extract text
            text = pytesseract.image_to_string(original_image)
            
            # Extract tables
            # tables = self._extract_tables()
            
            # Extract pictures/figures
            pictures = self._extract_pictures(image)
            
            # Determine if image is primarily a picture
            is_picture = len(text.strip()) < 50 
            
            # Extract metadata
            metadata = {
                'filename': os.path.basename(self.image_path),
                'format': original_image.format,
                'size': original_image.size,
                'mode': original_image.mode,
                'dpi': original_image.info.get('dpi'),
                'char_count': len(text),
                'word_count': len(text.split()),
                # 'table_count': len(tables) if tables else 0,
                'picture_count': len(pictures) if pictures else 0,
                'is_picture': is_picture,
                'source_type': 'image'
            }
            
            output_data = {
                'text': text if not is_picture else '',
                # 'tables': tables if not is_picture else [],
                'pictures': pictures if not is_picture else [self.image_path],
                'metadata': metadata
            }
            
            output_path = os.path.join(self.output_dir, 'image_content.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
                
            return output_data
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None

    def _extract_pictures(self, image) -> list:
        """Extract sub-images/figures from the main image."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            pictures = []
            for i, contour in enumerate(contours):
                # Filter small contours
                if cv2.contourArea(contour) > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    picture = image[y:y+h, x:x+w]
                    
                    # Save extracted picture
                    picture_path = os.path.join(self.output_dir, f'extracted_picture_{i}.png')
                    cv2.imwrite(picture_path, picture)
                    pictures.append(picture_path)
            
            return pictures
        except Exception as e:
            print(f"Error extracting pictures: {str(e)}")
            return []

    def _extract_tables(self) -> list:
        """Extract tables from image using Tesseract."""
        try:
            tables_data = pytesseract.image_to_data(Image.open(self.image_path), output_type=pytesseract.Output.DATAFRAME)
            
            tables = []
            if not tables_data.empty:
                for block_num, block_data in tables_data.groupby('block_num'):
                    if len(block_data) > 1:
                        table_dict = {
                            'table_id': f'table_{block_num}',
                            'content': block_data['text'].dropna().tolist()
                        }
                        tables.append(table_dict)
            return tables
        except Exception as e:
            print(f"Error extracting tables: {str(e)}")
            return []

    def preprocess_image(self) -> Optional[str]:
        """Preprocess image for better OCR results."""
        try:
            image = Image.open(self.image_path)
            image = image.convert('L')
            image = Image.fromarray(cv2.equalizeHist(np.array(image)))
            if max(image.size) > 3000:
                ratio = 3000 / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            output_path = os.path.join(self.output_dir, 'preprocessed_image.png')
            image.save(output_path)
            return output_path
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None
