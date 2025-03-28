import pandas as pd

def save_features_to_csv(features_list, output_csv):
    if not features_list:
        print("No features to save.")
        return
     
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(features_list)

    # Save DataFrame to CSV
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")