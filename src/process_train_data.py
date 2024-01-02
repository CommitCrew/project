import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def read_data(file_path):

    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):

    numerical_columns = ['Reading']
    scaler = MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns]) 
    return df

def split_data(df_preprocessed):
    train_percentage = 0.8

    # Calculate the index to split the data
    split_index = int(train_percentage * len(df_preprocessed))

    # Split the data into training and testing sets
    train_data = df_preprocessed.iloc[:split_index]
    test_data = df_preprocessed.iloc[split_index:]

    return train_data, test_data

if __name__ == '__main__':
    
        file_path = 'dummy_sensor_data.csv'
        df = read_data(file_path)
        print("Before preprocessing:")
        print(df.head(5))
        
        new_df = preprocess_data(df)
        print("After preprocessing:")
        print(new_df.head(5))
        
        train_data,test_data = split_data(new_df)
        print("Train Data Shape:", train_data.shape)
        print("Train Data:", train_data)
        print("Train Data Shape:", train_data.shape)
        print("Test Data:", test_data)