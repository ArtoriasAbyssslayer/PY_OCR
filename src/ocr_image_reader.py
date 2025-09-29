from tkinter import *
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import pytesseract
from pytesseract import Output
import re
from PIL import ImageFilter, ImageOps
import os
import cv2
import numpy as np

class OCRImageReader:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Image Reader")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Variables
        self.current_image_path = None
        self.current_image = None
        self.photo_image = None
        self.selected_language = StringVar(value='eng')
        self.auto_detect_lang = BooleanVar(value=False)
        # Preserve formatting (useful for code / tables)
        self.preserve_format_var = BooleanVar(value=False)
        
        # Available languages (common ones)
        self.languages = {
            'English': 'eng',
            'Greek': 'ell',
            'Spanish': 'spa',
            'French': 'fra',
            'German': 'deu',
            'Italian': 'ita',
            'Portuguese': 'por',
            'Russian': 'rus',
            'Chinese (Simplified)': 'chi_sim',
            'Chinese (Traditional)': 'chi_tra',
            'Japanese': 'jpn',
            'Korean': 'kor',
            'Arabic': 'ara',
            'Hindi': 'hin'
        }
        
        # Configure style
        self.setup_styles()
        
        # Create UI
        self.create_widgets()
        
    def setup_styles(self):
        """Setup modern dark theme with better styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Modern dark color scheme
        bg_dark = '#1e1e1e'
        bg_medium = '#252526'
        bg_light = '#2d2d30'
        bg_hover = '#3e3e42'
        fg_color = '#cccccc'
        fg_bright = '#ffffff'
        accent_color = '#0e639c'
        accent_hover = '#1177bb'
        accent_bright = '#007acc'
        border_color = '#3f3f46'
        
        # Configure root
        self.root.configure(bg=bg_dark)
        
        # Frame styles
        style.configure('TFrame', background=bg_dark)
        style.configure('Card.TFrame', background=bg_medium, relief=FLAT)
        
        # Label styles
        style.configure('Title.TLabel', 
                       font=('Segoe UI', 26, 'bold'), 
                       background=bg_dark, 
                       foreground=fg_bright)
        style.configure('TLabel', 
                       background=bg_dark, 
                       foreground=fg_color,
                       font=('Segoe UI', 10))
        style.configure('Card.TLabel', 
                       background=bg_medium, 
                       foreground=fg_color,
                       font=('Segoe UI', 10))
        
        # LabelFrame styles
        style.configure('TLabelframe', 
                       background=bg_medium, 
                       foreground=fg_color, 
                       borderwidth=0,
                       relief=FLAT)
        style.configure('TLabelframe.Label', 
                       background=bg_medium, 
                       foreground=fg_bright, 
                       font=('Segoe UI', 11, 'bold'),
                       padding=(5, 5))
        
        # Button styles - Modern flat design
        style.configure('TButton', 
                       background=bg_light,
                       foreground=fg_color,
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 10),
                       padding=(16, 10),
                       relief=FLAT)
        style.map('TButton',
                 background=[('active', bg_hover), ('pressed', bg_light)],
                 foreground=[('active', fg_bright)])
        
        # Primary button style
        style.configure('Primary.TButton',
                       background=accent_color,
                       foreground=fg_bright,
                       font=('Segoe UI', 11, 'bold'),
                       padding=(20, 12),
                       relief=FLAT)
        style.map('Primary.TButton',
                 background=[('active', accent_hover), ('pressed', accent_bright)])
        
        # Secondary button style
        style.configure('Secondary.TButton',
                       background=bg_light,
                       foreground=fg_color,
                       font=('Segoe UI', 10),
                       padding=(16, 10),
                       relief=FLAT)
        style.map('Secondary.TButton',
                 background=[('active', bg_hover)])
        
        # Entry style
        style.configure('TEntry',
                       fieldbackground=bg_light,
                       foreground=fg_color,
                       borderwidth=1,
                       bordercolor=border_color,
                       insertcolor=fg_bright,
                       relief=FLAT)
        
        # Combobox style
        style.configure('TCombobox',
                       fieldbackground=bg_light,
                       background=bg_light,
                       foreground=fg_color,
                       arrowcolor=fg_color,
                       borderwidth=1,
                       bordercolor=border_color,
                       relief=FLAT)
        style.map('TCombobox',
                 fieldbackground=[('readonly', bg_light)],
                 selectbackground=[('readonly', bg_light)])
        
        # Checkbutton style
        style.configure('TCheckbutton',
                       background=bg_medium,
                       foreground=fg_color,
                       font=('Segoe UI', 10))
        style.map('TCheckbutton',
                 background=[('active', bg_medium)])
        
        # Scrollbar style - minimal modern design
        style.configure('Vertical.TScrollbar',
                       background=bg_light,
                       troughcolor=bg_medium,
                       borderwidth=0,
                       arrowcolor=fg_color,
                       relief=FLAT)
        style.configure('Horizontal.TScrollbar',
                       background=bg_light,
                       troughcolor=bg_medium,
                       borderwidth=0,
                       arrowcolor=fg_color,
                       relief=FLAT)
        
    def create_widgets(self):
        """Create all UI widgets with responsive layout"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.grid(row=0, column=0, sticky=(N, S, E, W))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=3)
        main_container.columnconfigure(1, weight=2)
        main_container.rowconfigure(1, weight=1)
        
        # Left panel - Image preview
        left_panel = ttk.Frame(main_container, padding="20")
        left_panel.grid(row=0, column=0, rowspan=2, sticky=(N, S, E, W))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(1, weight=1)
        
        # Title section
        title_frame = ttk.Frame(left_panel)
        title_frame.grid(row=0, column=0, sticky=(E, W), pady=(0, 20))
        title_frame.columnconfigure(0, weight=1)
        
        title_label = ttk.Label(title_frame, text="OCR Image Reader", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, sticky=W)
        
        subtitle = ttk.Label(title_frame, 
                            text="Extract text from images with AI-powered OCR",
                            font=('Segoe UI', 10),
                            foreground='#888888')
        subtitle.grid(row=1, column=0, sticky=W, pady=(5, 0))
        
        # Image preview card
        preview_card = ttk.LabelFrame(left_panel, text="  Image Preview  ", 
                                      padding="20", style='TLabelframe')
        preview_card.grid(row=1, column=0, sticky=(N, S, E, W))
        preview_card.columnconfigure(0, weight=1)
        preview_card.rowconfigure(0, weight=1)
        
        # Canvas frame
        canvas_frame = ttk.Frame(preview_card)
        canvas_frame.grid(row=0, column=0, sticky=(N, S, E, W))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Canvas for image display
        self.canvas = Canvas(canvas_frame, bg='#2d2d30', relief=FLAT, 
                            borderwidth=0, highlightthickness=2,
                            highlightbackground='#007acc',
                            highlightcolor='#007acc')
        self.canvas.grid(row=0, column=0, sticky=(N, S, E, W))
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(canvas_frame, orient=VERTICAL, 
                                command=self.canvas.yview)
        v_scroll.grid(row=0, column=1, sticky=(N, S))
        h_scroll = ttk.Scrollbar(canvas_frame, orient=HORIZONTAL, 
                                command=self.canvas.xview)
        h_scroll.grid(row=1, column=0, sticky=(E, W))
        
        self.canvas.configure(yscrollcommand=v_scroll.set, 
                             xscrollcommand=h_scroll.set)
        
        # Placeholder
        self.canvas_text = self.canvas.create_text(
            300, 200, 
            text="📁\n\nNo image loaded\nClick 'Browse' to get started",
            font=('Segoe UI', 14), fill='#666666', justify=CENTER
        )
        
        self.canvas.bind('<Configure>', self.on_canvas_resize)
        
        # Right panel - Controls and output
        right_panel = ttk.Frame(main_container, padding="20 20 20 20")
        right_panel.grid(row=0, column=1, rowspan=2, sticky=(N, S, E, W))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(3, weight=1)
        
        # Input controls card
        input_card = ttk.LabelFrame(right_panel, text="  Input Settings  ", 
                                    padding="20", style='TLabelframe')
        input_card.grid(row=0, column=0, sticky=(E, W), pady=(0, 15))
        input_card.columnconfigure(0, weight=1)
        
        # File selection
        file_label = ttk.Label(input_card, text="Image File", style='Card.TLabel')
        file_label.grid(row=0, column=0, sticky=W, pady=(0, 8))
        
        file_frame = ttk.Frame(input_card)
        file_frame.grid(row=1, column=0, sticky=(E, W), pady=(0, 20))
        file_frame.columnconfigure(0, weight=1)
        
        self.path_entry = ttk.Entry(file_frame, state='readonly', 
                                     font=('Segoe UI', 9))
        self.path_entry.grid(row=0, column=0, sticky=(E, W), padx=(0, 10))
        
        browse_btn = ttk.Button(file_frame, text="📁 Browse", 
                               command=self.open_file,
                               style='Secondary.TButton')
        browse_btn.grid(row=0, column=1, sticky=(E, W))
        
        # Language selection
        lang_label = ttk.Label(input_card, text="OCR Language", style='Card.TLabel')
        lang_label.grid(row=2, column=0, sticky=W, pady=(0, 8))
        
        self.lang_combo = ttk.Combobox(input_card, 
                                       textvariable=self.selected_language,
                                       values=list(self.languages.keys()),
                                       state='readonly',
                                       font=('Segoe UI', 10))
        self.lang_combo.grid(row=3, column=0, sticky=(E, W), pady=(0, 12))
        self.lang_combo.set('English')
        
        self.auto_detect_check = ttk.Checkbutton(input_card, 
                                                 text="Auto-detect language",
                                                 variable=self.auto_detect_lang,
                                                 command=self.toggle_language_selection)
        self.auto_detect_check.grid(row=4, column=0, sticky=W)
        self.preserve_check = ttk.Checkbutton(input_card,
                                                 text="Preserve formatting (Code mode)",
                                                 variable=self.preserve_format_var,
                                                 command=self.on_preserve_toggle)
        self.preserve_check.grid(row=5, column=0, sticky=W, pady=(6,0))
        
        # Action buttons
        action_frame = ttk.Frame(right_panel)
        action_frame.grid(row=1, column=0, sticky=(E, W), pady=(0, 15))
        action_frame.columnconfigure(0, weight=1)
        
        self.read_btn = ttk.Button(action_frame, text="🔍 Extract Text", 
                                   command=self.read_image, 
                                   style='Primary.TButton', 
                                   state=DISABLED)
        self.read_btn.grid(row=0, column=0, sticky=(E, W), pady=(0, 10))
        
        # Secondary actions
        secondary_frame = ttk.Frame(action_frame)
        secondary_frame.grid(row=1, column=0, sticky=(E, W))
        secondary_frame.columnconfigure(0, weight=1)
        secondary_frame.columnconfigure(1, weight=1)
        secondary_frame.columnconfigure(2, weight=1)
        
        copy_btn = ttk.Button(secondary_frame, text="📋 Copy", 
                             command=self.copy_to_clipboard,
                             style='Secondary.TButton')
        copy_btn.grid(row=0, column=0, sticky=(E, W), padx=(0, 5))
        
        save_btn = ttk.Button(secondary_frame, text="💾 Save", 
                             command=self.save_text,
                             style='Secondary.TButton')
        save_btn.grid(row=0, column=1, sticky=(E, W), padx=(0, 5))
        
        clear_btn = ttk.Button(secondary_frame, text="🗑️ Clear", 
                              command=self.clear_all,
                              style='Secondary.TButton')
        clear_btn.grid(row=0, column=2, sticky=(E, W))
        
        # Output card
        output_card = ttk.LabelFrame(right_panel, text="  Extracted Text  ", 
                                     padding="20", style='TLabelframe')
        output_card.grid(row=3, column=0, sticky=(N, S, E, W))
        output_card.columnconfigure(0, weight=1)
        output_card.rowconfigure(0, weight=1)
        
        # Text output
        text_frame = ttk.Frame(output_card)
        text_frame.grid(row=0, column=0, sticky=(N, S, E, W))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        self.result_text = Text(text_frame, wrap=WORD, 
                               font=('Consolas', 10),
                               relief=FLAT, borderwidth=0,
                               bg='#1e1e1e', fg='#d4d4d4',
                               insertbackground='#ffffff',
                               selectbackground='#264f78',
                               selectforeground='#ffffff',
                               padx=10, pady=10)
        self.result_text.grid(row=0, column=0, sticky=(N, S, E, W))
        
        text_scroll = ttk.Scrollbar(text_frame, orient=VERTICAL, 
                                   command=self.result_text.yview)
        text_scroll.grid(row=0, column=1, sticky=(N, S))
        self.result_text.configure(yscrollcommand=text_scroll.set)
        # Horizontal scrollbar for preserving long lines (useful for code)
        h_text_scroll = ttk.Scrollbar(text_frame, orient=HORIZONTAL,
                                      command=self.result_text.xview)
        h_text_scroll.grid(row=1, column=0, sticky=(E, W))
        self.result_text.configure(xscrollcommand=h_text_scroll.set)
        
        # Status bar
        status_frame = ttk.Frame(right_panel, style='Card.TFrame')
        status_frame.grid(row=4, column=0, sticky=(E, W), pady=(15, 0))
        status_frame.columnconfigure(0, weight=1)
        
        self.status_var = StringVar(value="Ready to extract text from images")
        status_bar = ttk.Label(status_frame, textvariable=self.status_var,
                              font=('Segoe UI', 9), 
                              foreground='#888888',
                              padding=(10, 8))
        status_bar.grid(row=0, column=0, sticky=(E, W))
        
    def toggle_language_selection(self):
        """Toggle language combobox based on auto-detect checkbox"""
        if self.auto_detect_lang.get():
            self.lang_combo.configure(state='disabled')
        else:
            self.lang_combo.configure(state='readonly')
        
    def on_canvas_resize(self, event):
        """Handle canvas resize event to redraw image"""
        if self.current_image and event.width > 50 and event.height > 50:
            self.root.after(50, self.display_image)
        
    def open_file(self):
        """Open file dialog and load image"""
        filetypes = (
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("BMP files", "*.bmp"),
            ("All files", "*.*")
        )
        
        filename = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=filetypes
        )
        
        if filename:
            self.load_image(filename)
            
    def load_image(self, path):
        """Load and display image"""
        try:
            self.current_image_path = path
            self.current_image = Image.open(path)
            # Update path entry
            self.path_entry.configure(state='normal')
            self.path_entry.delete(0, END)
            self.path_entry.insert(0, path)
            self.path_entry.configure(state='readonly')
            
            # Display image
            self.display_image()
            
            # Enable read button
            self.read_btn.configure(state=NORMAL)
            
            # Update status
            filename = os.path.basename(path)
            size = self.current_image.size
            self.status_var.set(f"📷 {filename} • {size[0]}×{size[1]} pixels")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            self.status_var.set("❌ Error loading image")
            
    def display_image(self):
        """Display image on canvas with proper scaling"""
        if not self.current_image:
            return
            
        # Remove placeholder text if it exists
        try:
            self.canvas.delete(self.canvas_text)
        except:
            pass
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Minimum size check
        if canvas_width < 50 or canvas_height < 50:
            return
            
        img_width, img_height = self.current_image.size
        
        # Calculate scaling factor with padding
        padding = 40
        scale_w = (canvas_width - padding) / img_width
        scale_h = (canvas_height - padding) / img_height
        scale = min(scale_w, scale_h, 1.0)
        
        new_width = max(int(img_width * scale), 1)
        new_height = max(int(img_height * scale), 1)
        
        # Resize image
        resized_image = self.current_image.resize((new_width, new_height), 
                                                   Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo_image = ImageTk.PhotoImage(resized_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        
        # Center the image
        x = canvas_width // 2
        y = canvas_height // 2
        self.canvas.create_image(x, y, image=self.photo_image, anchor=CENTER)
        
        # Update scroll region
        self.canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))
        
    def detect_language(self, image):
        """Detect language from image using OSD (fallback to eng+ell if OSD fails)"""
        try:
            osd = pytesseract.image_to_osd(image)
            for line in osd.split('\n'):
                if 'Script:' in line:
                    script = line.split(':')[1].strip()
                    script_map = {
                        'Latin': 'eng+ell',   # fallback to both for mixed text
                        'Greek': 'ell',
                        'Cyrillic': 'rus',
                        'Arabic': 'ara',
                        'Han': 'chi_sim',
                        'Hangul': 'kor',
                        'Hiragana': 'jpn',
                        'Katakana': 'jpn'
                    }
                    lang = script_map.get(script)
                    if lang:
                        return lang
                    else:
                        print(f"Unknown script detected: {script}. Using eng+ell fallback.")
                        return 'eng+ell'
            print("No script detected. Using eng+ell fallback.")
            return 'eng+ell'
        except Exception as e:
            # Print the OSD error but still return a robust fallback
            print(f"OSD detection failed: {e}. Using eng+ell fallback.")
            return 'eng+ell'


    
    def is_probably_code(self, text):
        """Heuristic to decide if extracted text looks like source code."""
        if not text:
            return False
        indicators = ['#include', 'int main', 'std::', 'printf(', 'cout<<', '->', '::', '{', '}', ';', 'using namespace', 'template', 'class ']
        score = sum(text.count(tok) for tok in indicators)
        # Also count lines that end with semicolons
        semicolon_lines = sum(1 for line in text.splitlines() if line.strip().endswith(';'))
        return score >= 2 or semicolon_lines >= 2

    def reconstruct_text_from_data(self, data, code_mode=False):
        """Reconstruct text from pytesseract.image_to_data output while preserving spacing."""
        lines = {}
        n = len(data.get('text', []))
        for i in range(n):
            txt = data['text'][i]
            if not txt or not str(txt).strip():
                continue
            key = (int(data['block_num'][i]), int(data['par_num'][i]), int(data['line_num'][i]))
            lines.setdefault(key, {'words': [], 'top': int(data['top'][i])})
            lines[key]['words'].append({
                'text': txt,
                'left': int(data['left'][i]),
                'width': int(data['width'][i])
            })
            lines[key]['top'] = min(lines[key]['top'], int(data['top'][i]))

        sorted_lines = sorted(lines.items(), key=lambda kv: kv[1]['top'])

        # Compute global average char width
        total_width = 0
        total_chars = 0
        first_lefts = []
        for _, info in sorted_lines:
            words = sorted(info['words'], key=lambda w: w['left'])
            if not words:
                continue
            first_lefts.append(words[0]['left'])
            for w in words:
                total_width += max(1, w['width'])
                total_chars += max(1, len(w['text']))
        avg_char_width = max(1.0, total_width / max(1, total_chars))
        global_min_first_left = min(first_lefts) if first_lefts else 0

        out_lines = []
        for _, info in sorted_lines:
            words = sorted(info['words'], key=lambda w: w['left'])
            if not words:
                out_lines.append('')
                continue

            # leading spaces relative to leftmost text
            leading_pixels = max(0, words[0]['left'] - global_min_first_left)
            leading_spaces = int(round(leading_pixels / avg_char_width))
            parts = []
            if leading_spaces > 0:
                parts.append(' ' * leading_spaces)

            prev_right = words[0]['left'] + words[0]['width']
            parts.append(words[0]['text'])

            for w in words[1:]:
                gap = w['left'] - prev_right
                if gap <= 0:
                    parts.append(' ')
                else:
                    spaces = max(1, int(round(gap / avg_char_width)))
                    parts.append(' ' * spaces)
                parts.append(w['text'])
                prev_right = w['left'] + w['width']

            line_text = ''.join(parts).rstrip()
            out_lines.append(line_text)

        final_text = '\n'.join(out_lines)

        # Only collapse spaces for non-code mode
        if not code_mode:
            final_text = re.sub(r'[ \t]+', ' ', final_text)

        return final_text

    def on_preserve_toggle(self):
        """Adjust text widget wrap mode when user toggles preserve-formatting option."""
        try:
            if not self.preserve_format_var.get():
                text = re.sub(r'[ \t]+', ' ', text)
            if self.preserve_format_var.get():
                self.result_text.configure(wrap=NONE)
            else:
                self.result_text.configure(wrap=WORD)
        except Exception:
            pass


    def preprocess_image_for_ocr(self, pil_img, scale=2, code_mode=False):
        """
        Preprocess image to improve OCR accuracy for code/text.
        - Converts to grayscale
        - Resizes for better OCR
        - Enhances contrast
        - Binarizes adaptively
        - Optionally dilates thin strokes
        """

        # Convert to grayscale
        img = pil_img.convert('L')
        img_cv = np.array(img)

        # --- Step 1: Resize (scale up small fonts) ---
        if scale > 1:
            h, w = img_cv.shape
            img_cv = cv2.resize(img_cv, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        # --- Step 2: Adaptive thresholding (preserve fine symbols) ---
        img_cv = cv2.adaptiveThreshold(
            img_cv, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            31, 15
        )

        # --- Step 3: Invert if background is dark ---
        if np.mean(img_cv) < 127:
            img_cv = cv2.bitwise_not(img_cv)

        # --- Step 4: Enhance contrast ---
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_cv = clahe.apply(img_cv)

        # --- Step 5: Optional dilation for code symbols ---
        if code_mode:
            kernel = np.ones((2, 2), np.uint8)
            img_cv = cv2.morphologyEx(img_cv, cv2.MORPH_OPEN, kernel)


        # Convert back to PIL
        return Image.fromarray(img_cv)


    def split_code_blocks(self,pil_img):
        img = np.array(pil_img.convert('L'))
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Count non-white pixels per row
        row_sum = np.sum(thresh < 128, axis=1)

        # Rows with "almost no text" are separators
        separators = np.where(row_sum < np.max(row_sum) * 0.05)[0]

        blocks = []
        prev = 0
        for sep in separators:
            if sep - prev > 30:  # ignore tiny gaps
                block_img = pil_img.crop((0, prev, pil_img.width, sep))
                blocks.append(block_img)
            prev = sep
        if prev < pil_img.height:
            blocks.append(pil_img.crop((0, prev, pil_img.width, pil_img.height)))
        return blocks

    def clean_ocr_code(self,text: str) -> str:
        fixes = [
            (r"\baf[_]?\b", "if"),
            (r"= =", "=="),
            (r"- empty", ".empty()"),
            (r"- top[{]?\)?", ".top()"),
            (r"\bINT MAX\b", "INT_MAX"),
            (r"int - n", "int n"),
            (r"returndist", "return dist;"),
            (r"returnparent", "return parent"),
            (r"\bnjist\b", "n; i++"),
            (r"LZlnionbyrank", "// union by rank (fix manually)"),
            (r"priority_queuec", "priority_queue<"),
            (r"pair<int,int> >", "pair<int,int>>"),
        ]
        for pattern, repl in fixes:
            text = re.sub(pattern, repl, text)
        return text
    def auto_indent_cpp(self,code: str) -> str:
        lines = code.splitlines()
        indented = []
        level = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                indented.append("")
                continue

            # decrease indent before closing braces
            if stripped.startswith("}"):
                level = max(0, level - 1)

            indented.append("    " * level + stripped)

            # increase indent after opening braces
            if stripped.endswith("{"):
                level += 1

        return "\n".join(indented)



    def postprocess_code(self, text):
        """Fix common OCR mistakes in code"""
        if not text:
            return text

        # Normalize quotes/dashes
        text = text.replace('“','"').replace('”','"').replace('‘',"'").replace('’',"'")
        text = text.replace('–','-').replace('—','-')

        # Fix common OCR misreads
        text = re.sub(r'\bCoot\b', 'cout', text)
        text = re.sub(r'\bvectarcints\b', 'vector<int>', text)
        text = re.sub(r'\bpa\b', 'pair', text)
        text = re.sub(r'\bINT MAX\b', 'INT_MAX', text)

        # Fix pq-push / pq-pop
        text = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)-\s*(push|pop)\b', r'\1.\2', text)
        text = re.sub(r'priority\s+queue', 'priority_queue', text, flags=re.IGNORECASE)

        # Fix angle brackets
        text = re.sub(r'\s*<\s*', '<', text)
        text = re.sub(r'\s*>\s*', '>', text)

        # Normalize spacing around operators
        ops = [r'\+', '-', r'\*', '/', '%', '&', r'\|', r'\^', '<<', '>>', '=', '<', '>']

        for op in ops:
            text = re.sub(rf'\s*{op}\s*', f' {op} ', text)

        # Remove multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)

        # Fix trailing spaces
        text = '\n'.join(line.rstrip() for line in text.splitlines())
        
        text = self.clean_ocr_code(text)
        if(self.is_probably_code(text)):
            text = self.auto_indent_cpp(text)
        return text
    def copy_to_clipboard(self):
        """Copy the extracted text to clipboard."""
        text = self.result_text.get("1.0", "end-1c")
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_var.set("✅ Text copied to clipboard")
        else:
            self.status_var.set("⚠️ No text to copy")

    def save_text(self):
        """Save the extracted text to a file."""
        text = self.result_text.get("1.0", "end-1c")
        if not text:
            self.status_var.set("⚠️ No text to save")
            return

        from tkinter import filedialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                self.status_var.set(f"💾 Text saved to {file_path}")
            except Exception as e:
                self.status_var.set(f"❌ Failed to save file: {e}")

    def clear_all(self):
        """Clear the image and extracted text."""
        self.result_text.delete("1.0", "end")
        self.path_entry.configure(state='normal')
        self.path_entry.delete(0, 'end')
        self.path_entry.configure(state='readonly')
        self.canvas.delete("all")
        self.canvas_text = self.canvas.create_text(
            300, 200,
            text="📁\n\nNo image loaded\nClick 'Browse' to get started",
            font=('Segoe UI', 14), fill='#666666', justify='center'
        )
        self.current_image = None
        self.current_image_path = None
        self.read_btn.configure(state='disabled')
        self.status_var.set("Ready to extract text from images")

    def read_image(self):
        """Run OCR on the loaded image and display text."""
        if not self.current_image:
            self.status_var.set("⚠️ No image loaded")
            return

        self.status_var.set("🔍 Extracting text...")
        self.root.update_idletasks()
        code_mode = self.preserve_format_var.get()
        img = self.preprocess_image_for_ocr(self.current_image,code_mode)

        lang_code = 'eng'
        if self.auto_detect_lang.get():
            lang_code = self.detect_language(img)
        else:
            lang_name = self.lang_combo.get()
            lang_code = self.languages.get(lang_name, 'eng')

        # custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789{}[]();,.+-=*/&|<>_'
        custom_config = (
            r'--oem 3 --psm 6 '
            r'-c preserve_interword_spaces=1 '
            r'-c textord_min_linesize=2 '
            r'-c textord_space_size_is_variable=0 '
            r'-c textord_word_spacing=1 '
        )




        if code_mode:
            # Use image_to_data for preserving layout
            data = pytesseract.image_to_data(img, lang=lang_code, output_type=Output.DICT, config=custom_config)
            text = self.reconstruct_text_from_data(data, code_mode=code_mode)
            # Postprocess code-like text
            text = self.postprocess_code(text)
        
        else:
            text = pytesseract.image_to_string(img, lang=lang_code, config=custom_config)

        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.status_var.set("✅ Text extraction complete")
def main():
    import sys
    import os
    import ctypes
    from tkinter import Tk

    # Enable DPI awareness on Windows
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    root = Tk()

    # Determine icon path
    icon_path = None
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running as PyInstaller exe
        icon_path = os.path.join(sys._MEIPASS, 'ocr.ico')
    else:
        # Running as a script
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ocr.ico')

    if icon_path and os.path.exists(icon_path):
        try:
            root.iconbitmap(icon_path)
        except Exception:
            # silently fail if icon cannot be loaded
            pass

    app = OCRImageReader(root)
    root.mainloop()


if __name__ == "__main__":
    main()
