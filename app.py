from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import xgboost

app = Flask(__name__)

# 모델 불러오기 (훈련된 모델 파일이 있는 경우)
model = xgboost.XGBRegressor()
model.load_model("model.bst")

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = []
        for key in request.form.keys():
            value = request.form[key]
            if key in ['날짜', '요일', '날씨']:
                value = int(value)
            elif key in ['평균기온', '강수량']:
                value = float(value)
            input_data.append(value)

        input_data = np.array(input_data, dtype=np.float32).reshape(1, -1)
        prediction = model.predict(input_data)
        return render_template('index.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
