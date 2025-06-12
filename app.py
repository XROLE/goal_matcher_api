from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
import os

load_dotenv()

app = Flask(__name__)

# Set your Hugging Face API Token as an environment variable called HF_TOKEN
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Missing Hugging Face API token in environment variable 'HF_TOKEN'")

API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/sentence-similarity"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def get_similarity_matrix(goals):
    matrix = [[0.0 for _ in goals] for _ in goals]
    for i, goal in enumerate(goals):
        payload = {
            "inputs": {
                "source_sentence": goal,
                "sentences": goals
            }
        }
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=10)
            response.raise_for_status()
            similarities = response.json()
            matrix[i] = similarities
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error fetching similarity scores: {e}")
    return matrix

@app.route('/match-goals', methods=['POST'])
def match_goals():
    data = request.json

    if not data or 'users' not in data:
        return jsonify({"error": "Missing 'users' field in JSON body."}), 400

    users = data['users']
    if len(users) < 2:
        return jsonify({"error": "At least two users are required for matching."}), 400

    goals = [user['goal'] for user in users]
    try:
        sim_matrix = get_similarity_matrix(goals)
    except RuntimeError as err:
        return jsonify({"error": str(err)}), 500

    paired_indices = set()
    pairs = []

    for _ in range(len(goals) // 2):
        max_sim = -1
        pair = (-1, -1)
        for i in range(len(goals)):
            if i in paired_indices:
                continue
            for j in range(i + 1, len(goals)):
                if j in paired_indices:
                    continue
                if sim_matrix[i][j] > max_sim:
                    max_sim = sim_matrix[i][j]
                    pair = (i, j)
        if pair != (-1, -1):
            paired_indices.update(pair)
            pairs.append([users[pair[0]], users[pair[1]]])

    return jsonify({"pairs": pairs})

if __name__ == '__main__':
    app.run(debug=True)
