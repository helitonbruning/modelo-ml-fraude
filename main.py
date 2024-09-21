from flask import Flask, request, jsonify
import joblib
import numpy as np

# Carregar o modelo treinado
model = joblib.load('xgboost_model.pkl')

# Iniciar o app Flask
app = Flask(__name__)


# Definir rota para prever a classe
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Receber os dados em formato JSON
    features = np.array(data['features'])  # Extrair os dados do JSON

    # Fazer a previsão
    prediction = model.predict([features])

    # Retornar o resultado em JSON
    return jsonify({'prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)
