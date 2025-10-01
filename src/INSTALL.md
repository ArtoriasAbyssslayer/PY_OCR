# OCR Image Reader - Installation Guide

## Prerequisites

1. **Python 3.8 or higher** - Download from [python.org](https://www.python.org/downloads/)
2. **NVIDIA GPU with CUDA support** (optional, for GPU acceleration)
3. **Tesseract OCR** - Required for traditional OCR

### Installing Tesseract OCR

#### Windows
1. Download installer from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer and note the installation path (e.g., `C:\Program Files\Tesseract-OCR`)
3. Add Tesseract to your system PATH or set it in the code:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

#### Linux
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-ell
```

#### macOS
```bash
brew install tesseract
```

## Installation Steps

### 1. Clone or Download the Project

```bash
git clone <your-repo-url>
cd ocr-image-reader
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies

#### Option A: With CUDA 12.6 Support (GPU)

```bash
pip install -r requirements.txt
```

#### Option B: CPU Only

If you don't have an NVIDIA GPU or want to use CPU only:

```bash
# Install PyTorch CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install pytesseract Pillow opencv-python numpy transformers easyocr paddleocr python-doctr[torch] accelerate
```

### 4. Download Language Data (Optional)

For additional Tesseract languages:

```bash
# Windows: Download .traineddata files from
# https://github.com/tesseract-ocr/tessdata
# Place them in: C:\Program Files\Tesseract-OCR\tessdata\

# Linux:
sudo apt install tesseract-ocr-spa  # Spanish
sudo apt install tesseract-ocr-fra  # French
# etc.
```

### 5. Verify Installation

Test if everything is working:

```python
python dl_ocr.py
```

## Running the Application

```bash
python ocr_image_reader.py
```

## Troubleshooting

### PyTorch CUDA Issues

If PyTorch doesn't detect your GPU:

1. Check CUDA version:
   ```bash
   nvidia-smi
   ```

2. Install matching PyTorch version:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. Verify in Python:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print True
   print(torch.cuda.get_device_name(0))  # Should print your GPU name
   ```

### Tesseract Not Found

If you get "tesseract is not installed" error:

1. Verify Tesseract is installed:
   ```bash
   tesseract --version
   ```

2. If not in PATH, set it manually in `ocr_image_reader.py`:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

### EasyOCR Download Issues

EasyOCR downloads models on first use. If downloads fail:

1. Check your internet connection
2. Models are saved to `~/.EasyOCR/model/`
3. You can manually download models from [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)

### Memory Issues

If you run out of memory with deep learning models:

1. Close other applications
2. Use CPU mode: Set `use_gpu=False` in the code
3. Process smaller images
4. Use lighter models (EasyOCR instead of TrOCR)

## Model Storage

Deep learning models are cached in:
- Windows: `%USERPROFILE%\.cache\`
- Linux/macOS: `~/.cache/`

First-time model downloads may take several minutes.

## Additional Language Support

### EasyOCR Languages

EasyOCR supports 80+ languages. Common ones:
- English: 'en'
- Greek: 'el'
- Spanish: 'es'
- French: 'fr'
- German: 'de'
- Chinese: 'ch_sim', 'ch_tra'
- Japanese: 'ja'
- Korean: 'ko'

To add languages, modify in `dl_ocr.py`:
```python
self.model = easyocr.Reader(['en', 'el', 'fr'])  # Multiple languages
```

## Performance Tips

1. **GPU Acceleration**: Use CUDA for 5-10x faster processing
2. **Image Quality**: Higher resolution images work better
3. **Preprocessing**: The app includes automatic image enhancement
4. **Model Selection**:
   - **EasyOCR**: Best for multilingual, general use
   - **TrOCR**: Best for handwriting
   - **Tesseract**: Fastest, good for printed text
   - **PaddleOCR**: Good balance of speed/accuracy

## System Requirements

### Minimum
- CPU: Dual-core 2GHz+
- RAM: 4GB
- Storage: 2GB free space

### Recommended
- CPU: Quad-core 3GHz+
- RAM: 8GB+
- GPU: NVIDIA GPU with 4GB+ VRAM
- Storage: 5GB free space

## Support

For issues, check:
1. All dependencies are installed
2. Tesseract is in PATH
3. GPU drivers are up to date (for CUDA)
4. Python version is 3.8+

## License

[Your License Here]