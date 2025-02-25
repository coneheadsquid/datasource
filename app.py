from flask import Flask, request, jsonify, send_file
from celery import Celery
import os
import uuid
from gif_processor import extract_frames, process_frames, create_chunked_gifs  # Import functions

app = Flask(__name__)

# Configure Celery
app.config['CELERY_BROKER_URL'] = 'redis://redis:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://redis:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Temporary storage for uploaded and processed files
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/generate-gif', methods=['POST'])
def generate_gif():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded GIF
    input_gif_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_gif_path)

    # Generate a unique output path
    output_gif_path = os.path.join(OUTPUT_FOLDER, f"processed_{uuid.uuid4()}.gif")

    # Start the Celery task
    task = process_gif.delay(input_gif_path, output_gif_path)

    # Return the task ID for status checking
    return jsonify({"task_id": task.id}), 202

@celery.task
def process_gif(input_path, output_path):
    # Call the imported functions
    frames_folder = "raw"
    green_frames_folder = "green_raw"

    extract_frames(input_path, frames_folder)
    process_frames(frames_folder, green_frames_folder)
    create_chunked_gifs(green_frames_folder, output_path)

    return output_path

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)