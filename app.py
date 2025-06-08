from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/match-goals', methods=['POST'])
def match_goals():
    data = request.json

    if not data or 'users' not in data:
        return jsonify({"error": "Missing 'users' field in JSON body."}), 400

    users = data['users']
    if len(users) < 2:
        return jsonify({"error": "At least two users are required for matching."}), 400

    goals = [user['goal'] for user in users]
    embeddings = model.encode(goals)
    sim_matrix = cosine_similarity(embeddings)

    paired_indices = set()
    pairs = []

    for _ in range(len(goals) // 2):
        max_sim = -1
        pair = (-1, -1)
        for i in range(len(goals)):
            if i in paired_indices:
                continue
            for j in range(i+1, len(goals)):
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
