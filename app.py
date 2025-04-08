from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # To allow communication with frontend

@app.route('/', methods=['GET', 'POST'])
def process_string():
    if request.method == 'GET':
        input_string = request.args.get('input', '')
    else:
        data = request.get_json()
        input_string = data.get('input', '')

    # Example processing: reverse the string
    processed_string = input_string[::-1]

    return jsonify({'result': processed_string})

if __name__ == '__main__':
    app.run(port=8080)
