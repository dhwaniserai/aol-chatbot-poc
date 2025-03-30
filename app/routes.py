from flask import Blueprint, request, jsonify
from together import Together
from app.config import MODEL_NAME, TOGETHER_API_KEY
from app.vector_store import init_vectorstore

routes = Blueprint("routes", __name__)
vectorstore = init_vectorstore()
client = Together(api_key=TOGETHER_API_KEY)

@routes.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("query")
    if not user_input:
        return jsonify({"error": "Query missing"}), 400

    relevant_projects = vectorstore.similarity_search(user_input, k=3)

    projects_info = "\n\n".join([
        f"**{doc.metadata['title']}**\n- {doc.page_content}" for doc in relevant_projects
    ])

    prompt = f"""
You are an AI assistant for the **Art of Living**, dedicated to spreading peace, well-being, and service.

### INSTRUCTIONS
1️⃣ Recommend specific projects from the database.  
2️⃣ Match based on location, interests, and budget.  
3️⃣ For **donations**, only show projects within budget.  
4️⃣ If no exact match, suggest the closest options with a reason.  
5️⃣ For **volunteering**, match based on location and skills.  
6️⃣ Never invent projects.

### Relevant Projects:
{projects_info}

User Query: {user_input}
"""

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    full_response = stream.choices[0].message.content
    return jsonify({"response": full_response})
