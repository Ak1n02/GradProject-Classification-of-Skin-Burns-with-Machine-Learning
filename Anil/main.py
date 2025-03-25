import os
from feature_extraction import extract_features
from convert_to_csv import save_features_to_csv

def main():
    # Path to the dataset folder containing subfolders for different degrees
    dataset_folder = r'./Anil/dataset' 

    if not os.path.exists(dataset_folder):
        print(f"Dataset folder '{dataset_folder}' does not exist.")
        return

    # List of subfolders corresponding to each degree
    degree_folders = [folder for folder in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, folder))]
    if not degree_folders:
        print("no degree folders found")
        return
    # List to store the feature dictionaries
    features_list = []

    # Iterate through each degree folder
    for degree_folder in degree_folders:
        # Extract the degree value from the folder name
        try:
            degree = int(degree_folder.split('_')[-1])
        except ValueError:
            print(f"Skipping folder {degree_folder}: Invalid degree format.")
            continue
        
        degree_folder_path = os.path.join(dataset_folder, degree_folder)

        # Get all images in the degree folder
        image_paths = [os.path.join(degree_folder_path, f) for f in os.listdir(degree_folder_path) if f.endswith('.jpg') or f.endswith('.png')]
        # Iterate through each image and extract features
        for image_path in image_paths:
        
            # Extract features for each image
            features = extract_features(image_path, degree)
            if features:
                features_list.append(features)
            else:
                print(f"no features extracted")

    if features_list:
        # Save the extracted features to a CSV file
        save_features_to_csv(features_list, r'./Anil/dataset.csv')
    else:
        print("No features to save.")

if __name__ == "__main__":
    main()
    