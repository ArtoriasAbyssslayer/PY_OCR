import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import unicodedata


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
            error_msg = str(e)
            print(f"Error loading model: {error_msg}")
            
            # Provide helpful error messages
            if "Failed to resolve" in error_msg or "NameResolutionError" in error_msg:
                user_msg = "Network error: Cannot connect to model repository. Check your internet connection."
            elif "No module named" in error_msg:
                user_msg = f"Missing dependency. Install with: pip install {self.model_name}"
            else:
                user_msg = f"Error: {error_msg}"
            
            if progress_callback:
                progress_callback(user_msg)
            return False

    # ------------------- Loaders -------------------

    def _load_trocr(self, progress_callback=None):
        """Load Hugging Face TrOCR with better error handling"""
        model_name = (
            "microsoft/trocr-base-handwritten"
            if self.model_type == "handwritten"
            else "microsoft/trocr-base-printed"
        )
        
        if progress_callback:
            progress_callback(f"Loading TrOCR model: {model_name}")

        try:
            # Set environment variable to use standard HTTP instead of XET
            os.environ['HF_HUB_DISABLE_XET'] = '1'
            
            # Try to load with timeout and retry logic
            from huggingface_hub import snapshot_download
            
            if progress_callback:
                progress_callback("Downloading model files (this may take a few minutes)...")
            
            # Download model files first
            local_dir = snapshot_download(
                repo_id=model_name,
                cache_dir=str(self.cache_dir),
                resume_download=True,
                local_files_only=False
            )
            
            if progress_callback:
                progress_callback("Loading processor...")
            
            self.processor = TrOCRProcessor.from_pretrained(
                model_name, 
                cache_dir=str(self.cache_dir),
                local_files_only=True
            )
            
            if progress_callback:
                progress_callback("Loading model weights...")
            
            self.model = VisionEncoderDecoderModel.from_pretrained(
                model_name, 
                cache_dir=str(self.cache_dir),
                local_files_only=True
            )

            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            
            if progress_callback:
                progress_callback("TrOCR model loaded successfully!")
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "local_files_only" in error_msg or "offline mode" in error_msg:
                # Try again without local_files_only
                try:
                    if progress_callback:
                        progress_callback("Retrying download with fallback method...")
                    
                    self.processor = TrOCRProcessor.from_pretrained(
                        model_name, 
                        cache_dir=str(self.cache_dir),
                        force_download=False,
                        resume_download=True
                    )
                    
                    self.model = VisionEncoderDecoderModel.from_pretrained(
                        model_name, 
                        cache_dir=str(self.cache_dir),
                        force_download=False,
                        resume_download=True
                    )
                    
                    self.model.to(self.device)
                    self.model.eval()
                    self.is_loaded = True
                    return True
                except:
                    raise e
            else:
                raise e

    def _load_easyocr(self, progress_callback=None):
        if easyocr is None:
            raise ImportError("EasyOCR is not installed. Run `pip install easyocr`.")
        if progress_callback:
            progress_callback("Loading EasyOCR reader...")
        
        # EasyOCR downloads models automatically on first use
        model_storage_directory = os.path.join(str(self.cache_dir), 'easyocr')
        os.makedirs(model_storage_directory, exist_ok=True)
        
        self.model = easyocr.Reader(
            self.languages, 
            gpu=(self.device == 'cuda'),
            model_storage_directory=model_storage_directory
        )
        self.is_loaded = True
        
        if progress_callback:
            progress_callback("EasyOCR loaded successfully!")
        return True

    def _load_paddleocr(self, progress_callback=None):
        if PaddleOCR is None:
            raise ImportError("PaddleOCR is not installed. Run `pip install paddleocr`.")
        if progress_callback:
            progress_callback("Loading PaddleOCR...")
        
        lang = self.languages[0] if self.languages else 'en'
        self.model = PaddleOCR(
            use_angle_cls=True, 
            lang=lang, 
            use_gpu=(self.device == 'cuda'),
            show_log=False
        )
        self.is_loaded = True
        
        if progress_callback:
            progress_callback("PaddleOCR loaded successfully!")
        return True

    def _load_doctr(self, progress_callback=None):
        if ocr_predictor is None:
            raise ImportError("docTR is not installed. Run `pip install python-doctr[torch]`.")
        if progress_callback:
            progress_callback("Loading docTR...")
        
        self.model = ocr_predictor(pretrained=True)
        self.is_loaded = True
        
        if progress_callback:
            progress_callback("docTR loaded successfully!")
        return True

    # ------------------- Text Extraction -------------------
    def normalize_text(text):
        return unicodedata.normalize("NFC", text)
    def extract_text(self, image, languages=None):
        """
        Extract text from a PIL Image
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Allow runtime override of languages
        if languages is not None:
            self.languages = languages

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
        img_array = np.array(image)
        results = self.model.readtext(img_array)
        
        text_lines = [result[1] for result in results]
        return '\n'.join(text_lines)

    def _extract_paddleocr(self, image):
        """Extract text using PaddleOCR"""
        img_array = np.array(image)
        results = self.model.ocr(img_array, cls=True)
        
        text_lines = []
        if results and results[0]:
            for line in results[0]:
                if line and len(line) > 1:
                    text_lines.append(line[1][0])
        
        return '\n'.join(text_lines)

    def _extract_doctr(self, image):
        """Extract text using docTR"""
        img_array = np.array(image)
        result = self.model([img_array])
        
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


# Pre-download utility
def download_all_models(cache_dir=None, progress_callback=None):
    """
    Pre-download all available models
    
    Args:
        cache_dir: Directory to store models
        progress_callback: Function to call with progress updates
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    models_to_download = [
        ('trocr', 'printed', ['en']),
        ('trocr', 'handwritten', ['en']),
        ('easyocr', 'printed', ['en']),
    ]
    
    for model_name, model_type, languages in models_to_download:
        try:
            if progress_callback:
                progress_callback(f"Downloading {model_name} ({model_type})...")
            
            model = DLOCRModel(
                model_name=model_name,
                model_type=model_type,
                languages=languages,
                cache_dir=cache_dir
            )
            
            success = model.load_model(progress_callback)
            
            if success:
                model.unload_model()
                if progress_callback:
                    progress_callback(f"✓ {model_name} downloaded successfully")
            else:
                if progress_callback:
                    progress_callback(f"✗ Failed to download {model_name}")
                    
        except Exception as e:
            if progress_callback:
                progress_callback(f"✗ Error downloading {model_name}: {str(e)}")


# Convenience functions
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
    lang_code = self.languages.get(lang_name, 'eng')
    lang_map = {
                    'eng': 'en', 'ell': 'el', 'spa': 'es', 'fra': 'fr',
                    'deu': 'de', 'ita': 'it', 'por': 'pt', 'rus': 'ru',
                    'chi_sim': 'ch_sim', 'chi_tra': 'ch_tra',
                    'jpn': 'ja', 'kor': 'ko', 'ara': 'ar', 'hin': 'hi'
                }
    dl_lang = lang_map.get(lang_code, 'en')
    model = create_ocr_model(model_name, languages=[dl_lang])
    model.load_model()
    
    image = Image.open(image_path)
    text = model.extract_text(image,languages=[dl_lang])
    text = model.normalize_text(text)
    model.unload_model()
    return text


# Test function
if __name__ == "__main__":
    import sys
    
    print("Deep Learning OCR Module - Multi-Model Test")
    print("=" * 60)
    
    if '--download-all' in sys.argv:
        print("\nDownloading all models...")
        download_all_models(progress_callback=lambda msg: print(f"  {msg}"))
        print("\nAll models downloaded!")
        sys.exit(0)
    
    print("\nAvailable Models:")
    for name, desc in DLOCRModel.AVAILABLE_MODELS.items():
        print(f"  • {name}: {desc}")
    
    test_model = 'trocr'
    if len(sys.argv) > 1 and sys.argv[1] in DLOCRModel.AVAILABLE_MODELS:
        test_model = sys.argv[1]
    
    print(f"\n{'='*60}")
    print(f"Testing {test_model}...")
    print(f"{'='*60}\n")
    
    model = DLOCRModel(model_name=test_model, languages=['en','el'])
    
    success = model.load_model(progress_callback=lambda msg: print(f"  {msg}"))
    
    if success and len(sys.argv) > 2:
        image_path = sys.argv[2]
        print(f"\nProcessing: {image_path}")
        
        image = Image.open(image_path)
        text = model.extract_text(image)
        
        print("\nExtracted text:")
        print("-" * 60)
        print(text)
        print("-" * 60)
    
    model.unload_model()