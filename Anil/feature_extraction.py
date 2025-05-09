import cv2
import numpy as np
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def extract_features(image_path, degree):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Convert to L*a*b* color space
    lab_image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    #split L*, a*, b*  channels
    L, a, b = cv2.split(lab_image)

    # Remove zero a*,b* pairs to avoid division by zero in atan2
    valid_mask = (a != 0) | (b != 0)
    a = a[valid_mask]
    b = b[valid_mask]

    if len(a) == 0 or len(b) == 0:
        print(f"Skipping image {image_path}: No valid a* or b* values.")
        return None

     # Convert a* and b* to float32 and center around 0
    #default data type is uint8
    #convert to float32 for math calculations
    #-128 to change range from [0, 255] to [-128, 127]
    #a* greenish for negative reddish for positive
    #b* blueish for negative, yellowish for positive
    a_float = a.astype(np.float32) - 128
    b_float = b.astype(np.float32) - 128

    mean_a, std_a = np.mean(a_float), np.std(a_float)
    mean_b, std_b = np.mean(b_float), np.std(b_float)

    #calculate Hue angle in Degrees
    hue = np.arctan2(b_float, a_float) * (180 / np.pi)
    hue = np.mod(hue, 360) #keep it between [0, 360]
    hue_mean = np.mean(hue)
    hue_std = np.std(hue)

    #convert 2D array into 1D   
    #functions expects a 1D input
    a_kurtosis = kurtosis(a.flatten(), fisher=True)
    a_skewness = skew(a.flatten())
    b_kurtosis = kurtosis(b.flatten(), fisher=True)
    b_skewness = skew(b.flatten())

    # Entropy Features
    entropy_a = shannon_entropy(a)
    entropy_b = shannon_entropy(b)

    # Texture Features using GLCM
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # LBP Features
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)

    # Create dictionary
    features = {
        'degree': degree,
        'mean_a': mean_a, 'std_a': std_a, 'skew_a': a_skewness, 'kurtosis_a': a_kurtosis,
        'mean_b': mean_b, 'std_b': std_b, 'skew_b': b_skewness, 'kurtosis_b': b_kurtosis,
        'hue_mean': hue_mean, 'hue_std': hue_std,
        'entropy_a': entropy_a, 'entropy_b': entropy_b,
        'glcm_contrast': contrast, 'glcm_homogeneity': homogeneity,
        'glcm_energy': energy, 'glcm_correlation': correlation
    }

    return features

