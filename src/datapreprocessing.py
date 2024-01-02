#IMPORTING REQUIRED LIBRARIES
import pandas as pd
#--------------------------------------------------------------- DATA PRE PROCESSING -------------------------------------------------------

#THIS FUNCTION PERFORMS MEAN NORMALIZATION ON 'READING' COLUMN
def normalize_reading(reading):
    mean = reading.mean()
    std = reading.std()
    normalized_reading = (reading - mean) / std
    return normalized_reading

#THIS FUNCTION EXTRACTS THE MACHINE ID FROM THE STRING SO THAT IT IS A NUMERIC VALUE
def extractID(input_string):
    return input_string.apply(lambda x: int(''.join(filter(str.isdigit, str(x)))))

#READING CSV FILE CONTAINING GENERATED DATE INTO A DATAFRAME
dataframe = pd.read_csv('dummy_sensor_data.csv')

# TIME STAMP SPLIT INTO DAY, HOUR, MONTH
dataframe['Timestamp'] = pd.to_datetime(dataframe['Timestamp'])
dataframe['DAY'] = dataframe['Timestamp'].dt.day
dataframe['HOUR'] = dataframe['Timestamp'].dt.hour
dataframe['MONTH'] = dataframe['Timestamp'].dt.month

# MACHINE_ID STRING TO NUMERIC VALUE 
dataframe['NumericMachine_ID'] = extractID(dataframe['Machine_ID'])

# SENSOR_ID STRING TO NUMERIC VALUE  
dataframe['NumericSensor_ID'] = extractID(dataframe['Sensor_ID'])

# NORMALIZING READING VALUES
dataframe["NormalizedReading"] = normalize_reading(dataframe["Reading"])

# DROPPING COLUMNS NOT REQUIRED (USING THE NEWLY GENERATED PRE PROCESSED COLUMNS INSTEAD)
columns_to_drop = ['Timestamp', 'Machine_ID', 'Sensor_ID']
dataframe.drop(columns=columns_to_drop, inplace=True)

# NAMING THE PRE PROCESSED DATA FILE
preprocessed_file = 'preprocesseddata.csv'

# SAVING THE PREPROCESSED DATA PRESENT IN THE DATAFRAME TO A CSV FILE
dataframe.to_csv(preprocessed_file, index=False)


    