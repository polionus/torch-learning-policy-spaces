from PIL import Image, ImageDraw, ImageFont

def add_text_overlay(img: Image, label_text: str, font_size: int):
    if not label_text:
        return img
        
    # Ensure image is in RGB mode so colored text works
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    draw = ImageDraw.Draw(img)
    
    # 3. Font handling: Try to load a nice font, fallback to default if not found
    try:
        # You can change 'arial.ttf' to a specific path or font name
        # 15 is the font size
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default(size=font_size)
        
    x, y = 10, 10

    # 4. Draw the actual text in Red at (x, y)
    # Red is (255, 0, 0)
    draw.text((x, y), label_text, fill=(255, 0, 0), font=font)
    return img