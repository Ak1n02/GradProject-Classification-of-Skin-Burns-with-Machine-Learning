import cv2
import numpy as np
from scipy.stats import skew, kurtosis


def extract_features(image_path, degree):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Convert to L*a*b* color space
    lab_image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    #split L*, a*, b*  channels
    L, a, b = cv2.split(lab_image)

    mask = ~((L == 0) & (a == 128) & (b == 128))
    a_valid = a[mask]
    b_valid = b[mask]

    # Remove zero a*,b* pairs to avoid division by zero in atan2
    valid_mask = (a_valid != 0) | (b_valid != 0)
    a_valid = a_valid[valid_mask]
    b_valid = b_valid[valid_mask]

    if len(a_valid) == 0 or len(b_valid) == 0:
        print(f"Skipping image {image_path}: No valid a* or b* values.")
        return None

     # Convert a* and b* to float32 and center around 0
    #default data type is uint8
    #convert to float32 for math calculations
    #-128 to change range from [0, 255] to [-128, 127]
    #a* greenish for negative reddish for positive
    #b* blueish for negative, yellowish for positive
    a = a_valid.astype(np.float32) - 128
    b = b_valid.astype(np.float32) - 128

    mean_a, std_a = np.mean(a), np.std(a)
    mean_b, std_b = np.mean(b), np.std(b)

    #calculate Hue angle in Degrees
    hue = np.arctan2(b, a) * (180 / np.pi)
    hue = np.mod(hue, 360) #keep it between [0, 360]
    hue_mean = np.mean(hue)
    hue_std = np.std(hue)

    #convert 2D array into 1D   
    #functions expects a 1D input
    a_kurtosis = kurtosis(a.flatten(), fisher=True)
    a_skewness = skew(a.flatten())
    b_kurtosis = kurtosis(b.flatten(), fisher=True)
    b_skewness = skew(b.flatten())

    # Create a dictionary of extracted features with the degree label
    features = {
        'degree': degree,  # Add degree as Y value
        'mean_a': mean_a, 'std_a': std_a, 'skew_a': a_skewness, 'kurtosis_a': a_kurtosis,
        'mean_b': mean_b, 'std_b': std_b, 'skew_b': b_skewness, 'kurtosis_b': b_kurtosis,
        'hue_mean': hue_mean, 'hue_std': hue_std
    }

    return features

