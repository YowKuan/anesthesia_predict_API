from flask import Flask,redirect,url_for, render_template, request, send_from_directory, jsonify
import os
import json
import rfpca_model_prediction
import surgery_preprocess
import change_json_tocsv
import time

app = Flask(__name__, template_folder='templates')

@app.route('/api/json/userid', methods=['GET', 'POST'])
def api():
    current_time = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    filename = 'patientdata_{}.json'.format(current_time)
    content = request.get_json(force = True)
    print(content)
    with open(filename, 'w', encoding='utf8') as routefile:
        json.dump(content, routefile, ensure_ascii=False)
    change_json_tocsv.to_csv()
    surgery_preprocess.load_data()
    rfpca_model_prediction.prediction()
    with open('predict_result.json', newline='') as jsonfile:
        data = json.load(jsonfile)
        results = data["Result"]
    return data

if __name__ =='__main__':
    app.run(host='0.0.0.0', port='5000', debug = True)