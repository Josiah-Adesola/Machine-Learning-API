from flask import Flask, jsonify
import requests
import joblib

lr = joblib.load('model.pkl')
app = Flask(__name__)

@app.route("/predict", methods=['POST'])

def predict():
    if lr:
        try:
            json_ = request.json
            query_df = pd.DataFrame(json_)
            query = pd.get_dummies(query_df)
            
            prediction = lr.predict(query)
            return jsonify({'prediction': list(prediction)})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return ('No model here to use')
    
def hello():
    return "Welcome to Machine Learning model APIs"

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 5000
        
    lr = joblib.load('model.pkl')
    print('Model loaded')
    model_columns = joblib.load('model_columns.pkl')
    print('Model columns loaded')
    app.run(port=port, debug=True)