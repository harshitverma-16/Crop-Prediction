import numpy as np
import joblib
from flask import Flask, request, render_template, jsonify

model = joblib.load('crop_model.pkl')
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    features = np.array([float_feature])
    prediction = model.predict(features)
    predicted_crop = prediction[0].upper()
    return render_template('index.html', prediction_text=predicted_crop)




if __name__ == '__main__':
    app.run(port=3000, debug=True)