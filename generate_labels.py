from PIL import Image, ImageDraw, ImageFont
import os

# --- 1. CONFIGURATION (MUST MATCH YOUR DATA) ---
# NOTE: Using the Khmer script directly bypasses the rendering problem!
KHMER_LABELS = {
    'chumreapsour': 'ជម្រាបសួរ',
    'orkun': 'អរគុណ',
    'trov': 'ត្រឹមត្រូវ',
    'howru': 'សុខសប្បាយទេ',
    'mineyte': 'មិនអីទេ',
    'nothing': '...'
}

# Ensure these match your actual files and folder names
FONT_PATH = os.path.join('font', 'KhmerOS.ttf') 
OUTPUT_DIR = 'assets'
FONT_SIZE = 40
TEXT_COLOR = (255, 255, 255, 255) # White color with 100% opacity

# --- 2. GENERATION LOGIC ---
if not os.path.exists(FONT_PATH):
    print(f"Error: Font file not found at {FONT_PATH}. Please check the path.")
    exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

print("\n--- Generating Transparent KSL Labels ---")

for english_word, khmer_text in KHMER_LABELS.items():
    
    # Calculate text size using a dummy image (needed for Pillow)
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    
    # Measure text bounding box (requires Pillow 9+)
    try:
        bbox = draw.textbbox((0, 0), khmer_text, font=font)
    except AttributeError:
        # Fallback for older Pillow versions
        bbox = draw.textsize(khmer_text, font=font)
        bbox = (0, 0, bbox[0], bbox[1])
        
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Create canvas with padding
    padding = 10
    canvas_width = text_width + 2 * padding
    canvas_height = text_height + 2 * padding
    
    # Create the transparent image (RGBA: Alpha=0 for transparent)
    img = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw the text (using white color for visibility on the black bar)
    draw.text((padding, padding), khmer_text, font=font, fill=TEXT_COLOR) 

    # Save the file
    output_filename = os.path.join(OUTPUT_DIR, f'{english_word}.png')
    img.save(output_filename)
    
    print(f"✅ Generated: {output_filename} ({canvas_width}x{canvas_height})")

print("\n--- KSL Label generation complete. ---")