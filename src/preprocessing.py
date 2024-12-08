import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path, output_path):
    # Load raw data
    data = pd.read_csv(input_path)

    #Only standardized numeric features
    numerical_columns = data.select_dtypes(include=['float64']).columns

    #Normalization
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    data.drop("FCVC_minmax", axis = 1, inplace = True)

    data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    preprocess_data("../data/raw/obesity_dataset.csv", "../data/processed/obesity_standardized.csv")


