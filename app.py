from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from PIL import Image
import base64
import requests
import io

app = Flask(__name__)
app.secret_key = "change_this_to_any_random_secret_key"

# === Your OpenRouter API Key ===
OPENROUTER_API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# === 1. Image Captioning (using BLIP) ===
from transformers import BlipProcessor, BlipForConditionalGeneration
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def image_to_caption(image_path):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# === 2. Feedback Evaluation ===
def get_feedback(situation, answer):
    prompt = f"""
Situation meaning: {situation}
Student answer: {answer}

Your job:
Step 1: Determine relation using this rule set:
- If the answer shares at least one key idea (same subject or same action), output: "Related." and briefly state which idea matches and what is missing or different.
- If the answer shares no key ideas, output: "Not related." and state what the answer describes instead.

Step 2: Determine match quality:
- If all key ideas match, output: "This fully matches the situation."
- If some key ideas match but others are missing or different, output: "This partially matches the situation."
- If no key ideas match, output: "This does not match the situation."

Step 3: Rewrite the situation meaning in one short sentence starting with: "The situation shows ..."

Key definitions:
- Key ideas include subject (who/what), action (doing what), and important context (objects, location, etc.).
- Matching the subject OR the main action counts as shared key ideas.

Rules:
- Do NOT mention: caption, description, picture, image, scene, task, prompt, question, or writing.
- Do NOT praise or thank.
- Do NOT ask questions.
- Do NOT give suggestions.
- Do NOT use emojis or markdown.
- Use a neutral teacher-like tone.
- Keep each step exactly one sentence.

Output format strictly:

Related. or Not related. + reason,
This fully/partially/does not match the situation,
The situation shows ...

Output:
"""


    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    res = requests.post(BASE_URL, json=payload, headers=headers)
    text = res.json()["choices"][0]["message"]["content"].strip()
    return text


# === 3. Flask Routes ===

@app.route("/")
def choose():
    return render_template("choose.html")

@app.route("/answer/<image_id>")
def answer(image_id):
    return render_template("answer.html", image_id=image_id)

@app.route("/submit", methods=["POST"])
def submit():
    image_id = request.form.get("image_id")
    answer = request.form.get("answer")

    session["image_id"] = image_id
    session["answer"] = answer

    img_path = f"static/images/{image_id}.png"
    caption = image_to_caption(img_path)
    feedback = get_feedback(caption, answer)
    status = "related" if feedback.lower().startswith("related") else "not_related"

    session["feedback"] = feedback
    session["status"] = status

    return redirect(url_for("result"))

@app.route("/result")
def result():
    return render_template("result.html",
                           image_id=session.get("image_id"),
                           answer=session.get("answer"),
                           status=session.get("status"),
                           feedback=session.get("feedback"))

if __name__ == "__main__":
    app.run(debug=True)
