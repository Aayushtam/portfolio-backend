from flask import Flask, request, jsonify
from flask_cors import CORS
import personal_assistant as pa


app = Flask(__name__)
CORS(app)

# Ensure the agent is created on startup (lazy creation is fine too)
agent = pa.get_agent()


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'error': 'no message provided'}), 400
    try:
        reply = pa.respond_to_query(message)
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': 'assistant error', 'details': str(e)}), 500


if __name__ == '__main__':
    # Run on port 7860 by default; adjust if needed
    app.run(host='0.0.0.0', port=7860)
