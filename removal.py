import os
import onnxruntime as ort
import numpy as np
import cv2
from rembg import remove
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Load ONNX model
model_path = "u2net_human_seg.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


# Preprocess image
def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size  # Save original size for later
    image = image.resize((320, 320))  # Resize for U-2-Net
    image = np.array(image).astype(np.float32) / 255.0  # Normalize
    image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
    image = np.expand_dims(image, axis=0)  # Add batch dim
    return image, orig_size


# Postprocess mask
def postprocess(output, orig_size):
    mask = output.squeeze()  # Remove batch dimension
    mask = cv2.resize(mask, orig_size)  # Resize back to original size
    mask = (mask > 0.5).astype(np.uint8) * 255  # Convert to binary mask
    return mask


# Remove background using U-2-Net
def remove_background_u2net(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    orig_h, orig_w = image.shape[:2]

    preprocessed_img, orig_size = preprocess(image_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: preprocessed_img})[0]
    mask = postprocess(output, orig_size)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(image, mask)
    result_rgba = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result_rgba[:, :, 3] = mask[:, :, 0]  # Alpha channel from mask

    # Convert BGR to RGB
    result_rgb = cv2.cvtColor(result_rgba, cv2.COLOR_BGRA2RGBA)

    return result_rgb

# Remove background using rembg
def remove_background_rembg(image_path):
    input_image = Image.open(image_path)
    output_image = remove(input_image)
    return np.array(output_image)


# Compare results using SSIM
def compare_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2GRAY)
    return ssim(gray1, gray2)


# Foreground pixel analysis
def foreground_pixel_analysis(output_img):
    alpha_channel = output_img[:, :, 3]
    return np.count_nonzero(alpha_channel) / alpha_channel.size


# Edge detection similarity
def edge_similarity(img1, img2):
    edges1 = cv2.Canny(cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY), 100, 200)
    edges2 = cv2.Canny(cv2.cvtColor(img2, cv2.COLOR_BGRA2GRAY), 100, 200)
    return ssim(edges1, edges2)


# Histogram difference
def histogram_difference(img1, img2):
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


# Decide the best method
def decide_best_method(output_u2net, output_rembg):
    score_ssim = compare_images(output_u2net, output_rembg)
    score_foreground = foreground_pixel_analysis(output_u2net)
    score_edge = edge_similarity(output_u2net, output_rembg)
    score_hist = histogram_difference(output_u2net, output_rembg)

    if(score_foreground<0.2):
        return (output_rembg, "rembg")

    u2net_score = (score_ssim + score_foreground + score_edge + score_hist) / 4
    rembg_score = (1 - score_ssim + (1 - score_foreground) + (1 - score_edge) + (1 - score_hist)) / 4

    return (output_u2net, "U-2-Net") if u2net_score > rembg_score else (output_rembg, "rembg")


# Process and choose the best result
def process_and_choose_best(image_path, output_path):
    output_u2net = remove_background_u2net(image_path)
    output_rembg = remove_background_rembg(image_path)

    best_output, best_method = decide_best_method(output_u2net, output_rembg)

    # Convert the result to a PIL image (RGBA)
    best_output_pil = Image.fromarray(best_output)

    # Create a white background image
    white_bg = Image.new("RGB", (best_output.shape[1], best_output.shape[0]), (255, 255, 255))
    white_bg.paste(best_output_pil, (0, 0), best_output_pil)

    # Save the result as PNG
    white_bg.save(output_path, "PNG")
    print(f"Processed: {image_path} → {output_path} (Best Method: {best_method})")

# Process all images in a directory
def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_extensions = (".jpg", ".jpeg", ".png")
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")
            process_and_choose_best(input_path, output_path)


# Example Usage
if __name__ == "__main__":
    input_folder = "../Third_Degre_Extra"  # Change this to your input directory
    output_folder = "../Third_Degree_ExtraRB"  # Change this to your output directory
    process_directory(input_folder, output_folder)