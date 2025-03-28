!pip install flask flask-cors pyngrok

from pyngrok import ngrok
import os
import torch
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS  # Import CORS
from diffusers import StableDiffusionPipeline
from huggingface_hub import snapshot_download




# ✅ Initialize Flask App
app = Flask(__name__)
#CORS(app)  # Enable CORS for all routes
# Allow both localhost and your deployed Vercel site
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://myapp-lac-nine.vercel.app"]}})

# ✅ Define Model Path
MODEL_DIR = "saved_model"
OUTPUT_IMAGE = "generated_image.png"

# ✅ Load Model from Disk
if os.path.exists(MODEL_DIR):
    print("🔄 Loading model from local storage...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16
    ).to("cuda")
    print("✅ Model loaded successfully!")
else:
    raise FileNotFoundError("❌ Model not found! Please save the model first.")

# ✅ Image Generation Function
def generate_sd_image(prompt, num_steps=100, guidance=10):
    """Generates an image from text and saves it."""
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance
    ).images[0]

    image.save(OUTPUT_IMAGE)
    return OUTPUT_IMAGE

# ✅ API Endpoint to Generate Image
@app.route('/generate', methods=['POST', 'OPTIONS'])  # Add OPTIONS method for CORS preflight
def generate():
    """API Endpoint to generate an image from text prompt."""
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({"message": "Preflight request successful"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    data = request.json
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "❌ No prompt provided!"}), 400

    try:
        image_path = generate_sd_image(prompt)

        # Read the image and convert to base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        return jsonify({
            "image": f"data:image/png;base64,{encoded_string}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Additional imports for base64 encoding
import base64

# Set your ngrok authtoken
ngrok.set_auth_token("2unN6AH1fkn0QZzvNrlLlh7utJh_3Pg9vNdiYec3Htgzbs7Dz")

# Create public URL
public_url = ngrok.connect(5000).public_url
print(f"🚀 Public API URL: {public_url}")

# Run the app
app.run(port=5000)
