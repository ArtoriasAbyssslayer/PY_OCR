import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np

# Hugging Face
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Other OCR backends (import if installed)
try:
    import easyocr
except ImportError:
    easyocr = None

try:
    from doctr.models import ocr_predictor
    from doctr.io import DocumentFile
except ImportError:
    ocr_predictor = None
    DocumentFile = None

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None


class DLOCRModel:
    """Deep Learning OCR wrapper supporting multiple backends"""

    AVAILABLE_MODELS = {
        'trocr': 'TrOCR (Transformer, good for handwriting/printed)',
        'easyocr': 'EasyOCR (80+ languages, fast)',
        'paddleocr': 'PaddleOCR (Lightweight, very fast)',
        'doctr': 'docTR (End-to-end, accurate)',
    }

    def __init__(self, model_name='easyocr', model_type='printed', languages=None, cache_dir=None):
        """
        Args:
            model_name (str): Which OCR engine to use ('trocr','easyocr','paddleocr','doctr')
            model_type (str): 'printed' or 'handwritten' (only relevant for TrOCR)
            languages (list): List of language codes (e.g., ['en', 'el'])
            cache_dir (str): Directory to cache model weights (default: ./models)
        """
        self.model_name = model_name.lower()
        self.model_type = model_type.lower()
        self.languages = languages or ['en']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Cache dir
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), 'models')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.processor = None
        self.model = None
        self.is_loaded = False

        print(f"OCR initialized with backend={self.model_name} on device={self.device}")

    def load_model(self, progress_callback=None):
        """Load selected OCR model"""
        try:
            if self.model_name == 'trocr':
                return self._load_trocr(progress_callback)
            elif self.model_name == 'easyocr':
                return self._load_easyocr(progress_callback)
            elif self.model_name == 'paddleocr':
                return self._load_paddleocr(progress_callback)
            elif self.model_name == 'doctr':
                return self._load_doctr(progress_callback)
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            if progress_callback:
                progress_callback(f"Error: {e}")
            return False

    # ------------------- Loaders -------------------

    def _load_trocr(self, progress_callback=None):
        """Load Hugging Face TrOCR"""
        model_name = (
            "microsoft/trocr-base-handwritten"
            if self.model_type == "handwritten"
            else "microsoft/trocr-base-printed"
        )
        if progress_callback:
            progress_callback(f"Loading TrOCR model: {model_name}")

        self.processor = TrOCRProcessor.from_pretrained(
            model_name, cache_dir=str(self.cache_dir)
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_name, cache_dir=str(self.cache_dir)
        )

        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        return True

    def _load_easyocr(self, progress_callback=None):
        if easyocr is None:
            raise ImportError("EasyOCR is not installed. Run `pip install easyocr`.")
        if progress_callback:
            progress_callback("Loading EasyOCR reader...")
        self.model = easyocr.Reader(self.languages, gpu=(self.device == 'cuda'))
        self.is_loaded = True
        return True

    def _load_paddleocr(self, progress_callback=None):
        if PaddleOCR is None:
            raise ImportError("PaddleOCR is not installed. Run `pip install paddleocr`.")
        if progress_callback:
            progress_callback("Loading PaddleOCR...")
        # Map first language or default to 'en'
        lang = self.languages[0] if self.languages else 'en'
        self.model = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=(self.device == 'cuda'))
        self.is_loaded = True
        return True

    def _load_doctr(self, progress_callback=None):
        if ocr_predictor is None:
            raise ImportError("docTR is not installed. Run `pip install python-doctr[torch]`.")
        if progress_callback:
            progress_callback("Loading docTR...")
        self.model = ocr_predictor(pretrained=True)
        self.is_loaded = True
        return True

    # ------------------- Text Extraction -------------------

    def extract_text(self, image):
        """
        Extract text from a PIL Image
        
        Args:
            image: PIL Image object
            
        Returns:
            str: Extracted text
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.model_name == 'trocr':
            return self._extract_trocr(image)
        elif self.model_name == 'easyocr':
            return self._extract_easyocr(image)
        elif self.model_name == 'paddleocr':
            return self._extract_paddleocr(image)
        elif self.model_name == 'doctr':
            return self._extract_doctr(image)

    def _extract_trocr(self, image):
        """Extract text using TrOCR"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text

    def _extract_easyocr(self, image):
        """Extract text using EasyOCR"""
        # Convert PIL to numpy array
        img_array = np.array(image)
        results = self.model.readtext(img_array)
        
        # Extract text from results
        text_lines = [result[1] for result in results]
        return '\n'.join(text_lines)

    def _extract_paddleocr(self, image):
        """Extract text using PaddleOCR"""
        # Convert PIL to numpy array
        img_array = np.array(image)
        results = self.model.ocr(img_array, cls=True)
        
        # Extract text from results
        text_lines = []
        if results and results[0]:
            for line in results[0]:
                if line and len(line) > 1:
                    text_lines.append(line[1][0])
        
        return '\n'.join(text_lines)

    def _extract_doctr(self, image):
        """Extract text using docTR"""
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Process with docTR
        result = self.model([img_array])
        
        # Extract text
        text_lines = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = ' '.join([word.value for word in line.words])
                    text_lines.append(line_text)
        
        return '\n'.join(text_lines)

    def process_full_image(self, image, detect_lines_auto=True):
        """
        Process full image with optional line detection
        
        Args:
            image: PIL Image
            detect_lines_auto: Whether to automatically detect lines (for TrOCR)
            
        Returns:
            str: Extracted text
        """
        return self.extract_text(image)

    def unload_model(self):
        """Unload model to free memory"""
        self.model = None
        self.processor = None
        self.is_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Model unloaded and memory cleared")


# Convenience function
def create_ocr_model(model_name='easyocr', languages=None, cache_dir=None):
    """
    Factory function to create OCR model
    
    Args:
        model_name: 'trocr', 'easyocr', 'paddleocr', or 'doctr'
        languages: List of language codes (e.g., ['en', 'el'])
        cache_dir: Directory to cache models
        
    Returns:
        DLOCRModel instance
    """
    languages = languages or ['en']
    model = DLOCRModel(model_name=model_name, languages=languages, cache_dir=cache_dir)
    return model


def quick_ocr(image_path, model_name='easyocr', languages=None):
    """
    Quick OCR on an image file
    
    Args:
        image_path: Path to image file
        model_name: Model to use
        languages: Languages to support
        
    Returns:
        str: Extracted text
    """
    languages = languages or ['en']
    model = create_ocr_model(model_name, languages)
    model.load_model()
    
    image = Image.open(image_path)
    text = model.extract_text(image)
    
    model.unload_model()
    return text


# Test function
if __name__ == "__main__":
    import sys
    
    print("Deep Learning OCR Module - Multi-Model Test")
    print("=" * 60)
    print("\nAvailable Models:")
    for name, desc in DLOCRModel.AVAILABLE_MODELS.items():
        print(f"  • {name}: {desc}")
    
    # Test with first available model
    test_model = 'easyocr'  # Change this to test different models
    
    print(f"\n{'='*60}")
    print(f"Testing {test_model}...")
    print(f"{'='*60}\n")
    
    model = DLOCRModel(model_name=test_model, languages=['en'])
    
    success = model.load_model(progress_callback=lambda msg: print(f"  {msg}"))
    
    if success and len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nProcessing: {image_path}")
        
        image = Image.open(image_path)
        text = model.extract_text(image)
        
        print("\nExtracted text:")
        print("-" * 60)
        print(text)
        print("-" * 60)
    
    model.unload_model()