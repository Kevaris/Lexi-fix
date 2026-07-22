from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests securely

# --- SERVER CONFIGURATION ---
LLAMA_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLAMA_API_KEY = os.environ.get("GROQ_API_KEY")

ANALYZER_SYSTEM_PROMPT = """You are LexiFix, an expert multi-language code corrector and tutor.
Your task is to analyze raw source code provided by a student in a specific language (Python, Java, or HTML).

INSTRUCTIONS:
1. Check the entire code snippet for syntax errors, missing brackets, typos, improper indentation, or bad tags.
2. If there are syntax errors, fix them completely.
3. If the code is already 100% correct, keep it as is.
4. Provide a line-by-line or section-by-section breakdown explaining what every line does in simple, plain English. If any changes were made to fix errors, explicitly explain what was wrong and why it was changed.

OUTPUT FORMAT REQUIREMENTS:
You MUST return your final response as a strict, valid JSON object with EXACTLY these three keys:
{
  "has_errors": true or false,
  "corrected_code": "The fully corrected, working source code string",
  "explanation": "Clear, line-by-line breakdown and summary of changes made in plain English (formatted with clean line breaks)"
}

Do NOT include markdown formatting (like ```json) outside the JSON object. Return raw JSON only."""

HELP_SYSTEM_PROMPT = """You are the LexiFix Help Assistant, a friendly and concise AI tutor designed to help computer science students with programming, syntax, and logic questions. Keep answers clear, encouraging, and brief."""


@app.route('/analyze', methods=['POST'])
def analyze_code():
    data = request.json or {}
    language = data.get("language", "python").strip()
    user_code = data.get("code", "").strip()

    if not user_code:
        return jsonify({"error": "No source code provided."}), 400

    prompt_content = f"Language: {language}\n\nSource Code:\n{user_code}"

    headers = {
        "Authorization": f"Bearer {LLAMA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_content}
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post(LLAMA_API_URL, json=payload, headers=headers, timeout=15)
        response_json = response.json()

        if 'error' in response_json:
            return jsonify({"error": f"Groq Engine Error: {response_json['error'].get('message', 'Access Restricted')}"}), 400

        if 'choices' not in response_json or not response_json['choices']:
            return jsonify({"error": "Invalid engine response payload received."}), 500

        result_text = response_json['choices'][0]['message']['content'].strip()
        parsed_result = json.loads(result_text)

        return jsonify({
            "has_errors": parsed_result.get("has_errors", False),
            "corrected_code": parsed_result.get("corrected_code", user_code),
            "explanation": parsed_result.get("explanation", "Analysis complete.")
        })

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse analysis payload from engine."}), 500
    except Exception as e:
        return jsonify({"error": f"Internal Analysis Core Error: {str(e)}"}), 500


@app.route('/help', methods=['POST'])
def help_assistant():
    data = request.json or {}
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Message body is empty."}), 400

    headers = {
        "Authorization": f"Bearer {LLAMA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": HELP_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.5
    }

    try:
        response = requests.post(LLAMA_API_URL, json=payload, headers=headers, timeout=10)
        response_json = response.json()

        if 'error' in response_json:
            return jsonify({"error": f"Groq Engine Error: {response_json['error'].get('message', 'Access Restricted')}"}), 400

        reply_text = response_json['choices'][0]['message']['content'].strip()
        return jsonify({"reply": reply_text})

    except Exception as e:
        return jsonify({"error": f"Internal Help Core Error: {str(e)}"}), 500


@app.route('/healthz')
def health_check():
    return "LexiFix core is awake!", 200


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
