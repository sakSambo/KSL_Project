import tkinter as tk
from tkinter import filedialog, font as tkfont
from PIL import Image, ImageDraw, ImageFont
import os

# --- CONFIGURATION (Must match your project structure) ---
FONT_PATH = os.path.join('font', 'KhmerOS.ttf')
OUTPUT_DIR = 'assets'
FALLBACK_TEXT = "Type Khmer text here..."
FONT_SIZE = 40
TEXT_COLOR = (255, 255, 255, 255) # White text for the black subtitle bar

if not os.path.exists(FONT_PATH):
    print(f"Error: Font file not found at {FONT_PATH}. Please check the 'font/' folder.")
    exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)
try:
    # Attempt to load the font for Pillow
    pil_font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except IOError:
    print(f"Error loading PIL font. Check font file integrity at {FONT_PATH}.")
    exit()


def save_as_png():
    """Reads text from the entry box, draws it perfectly, and saves as transparent PNG."""
    khmer_text = text_entry.get().strip()
    english_name = name_entry.get().strip().replace(' ', '_').lower()

    if not khmer_text or not english_name:
        status_label.config(text="⚠️ Please enter both Khmer Text and English Name.", fg='red')
        return

    # 1. Measure text dimensions (using PIL's bounding box logic)
    # We rely on Tkinter for *rendering*, but PIL for *saving*.
    draw_temp = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    try:
        bbox = draw_temp.textbbox((0, 0), khmer_text, font=pil_font)
    except AttributeError:
        # Fallback for older PIL versions
        bbox = draw_temp.textsize(khmer_text, font=pil_font)
        bbox = (0, 0, bbox[0], bbox[1])
    
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Add padding
    padding = 10
    canvas_width = text_width + 2 * padding
    canvas_height = text_height + 2 * padding
    
    # 2. Create the transparent image (RGBA: Alpha=0 for transparent background)
    img = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 3. Draw the text onto the transparent canvas
    draw.text((padding, padding), khmer_text, font=pil_font, fill=TEXT_COLOR) 

    # 4. Save the file
    output_filename = os.path.join(OUTPUT_DIR, f'{english_name}.png')
    img.save(output_filename)
    
    status_label.config(text=f"✅ Saved! {english_name}.png ({canvas_width}x{canvas_height})", fg='green')
    print(f"Generated: {output_filename}")


# --- TKINTER GUI SETUP ---
root = tk.Tk()
root.title("KSL Label Generator")

# Use a Tkinter Font object linked to the Khmer font for reliable display
try:
    display_font = tkfont.Font(family="Khmer OS", size=FONT_SIZE, name="KhmerDisplay")
except Exception as e:
    print(f"Could not load system font for preview. Using default. Error: {e}")
    display_font = tkfont.Font(size=FONT_SIZE)


# --- WIDGETS ---

# 1. Khmer Text Input
tk.Label(root, text="Khmer Text (ជម្រាបសួរ):", pady=10).pack()
text_entry = tk.Entry(root, width=40, font=('Arial', 14))
text_entry.insert(0, FALLBACK_TEXT)
text_entry.pack(padx=10)

# 2. English Name Input
tk.Label(root, text="English Name (for Python/File):", pady=10).pack()
name_entry = tk.Entry(root, width=40)
name_entry.insert(0, "e.g., howru")
name_entry.pack(padx=10)

# 3. Live Preview (Uses native OS rendering for CTL)
tk.Label(root, text="Live Preview:", pady=10).pack()
preview_label = tk.Label(root, text="[Your Text Here]", font=display_font, bg='black', fg='white', padx=20, pady=20)
preview_label.pack(pady=10)

def update_preview(event=None):
    text = text_entry.get()
    if text:
        preview_label.config(text=text)
    else:
        preview_label.config(text="[Empty]")
    
# Bind update function to key press in the text entry field
text_entry.bind('<KeyRelease>', update_preview)

# 4. Save Button
save_button = tk.Button(root, text="Generate and Save PNG Label", command=save_as_png, bg='blue', fg='white', font=('Arial', 14, 'bold'))
save_button.pack(pady=20)

# 5. Status Label
status_label = tk.Label(root, text="Ready.", fg='gray')
status_label.pack(pady=5)

# --- RUN ---
root.mainloop()