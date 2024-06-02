# server.py
from flask import Flask, render_template, request, jsonify
from main.main import evaluate_message  # Adjusted import statement

app = Flask(__name__)

# Allows for the interaction between server and front-end (interface)
@app.route('/')
def home():
    # Render the template for the chat interface
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    message = data.get('message', '')

    # Call the function from main.py to evaluate the message
    result = evaluate_message(message)

    # Convert the result to a JSON serializable format
    serialized_result = str(result)

    return jsonify({'result': serialized_result})

if __name__ == '__main__':
    app.run(debug=True)