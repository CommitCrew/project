from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('app.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file,encoding='utf-8')
        columns_to_drop = ['NormalizedReading']
        df.drop(columns=columns_to_drop, inplace=True)
        predictions = model.predict(df)
        print(predictions)
         # Combine Machine_ID and predictions into a list of tuples
        combined_data = list(zip(df['NumericMachine_ID'],df['NumericSensor_ID'], predictions.tolist()))

        return render_template('app.html', combined_data=combined_data)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
