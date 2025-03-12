from rembg import remove
from PIL import Image

inputp = "img_2.png"
outputp = "img2nobg_white.png"

input_image = Image.open(inputp)
output_image = remove(input_image)

# Create a white background image
white_bg = Image.new("RGB", output_image.size, (255, 255, 255))
white_bg.paste(output_image, (0, 0), output_image)
white_bg.save(outputp, "PNG")