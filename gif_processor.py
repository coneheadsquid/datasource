# gif_processor.py
import cv2
import numpy as np
from PIL import Image, ImageSequence, ImageEnhance
import os
import random

# Global state to track falling character positions and opacity
character_positions = {}
character_opacities = {}
character_values = {}

def extract_frames(video_path, frames_folder):
    os.makedirs(frames_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {frames_folder}")

def initialize_character_positions(width, height):
    global character_positions, character_opacities, character_values
    column_step = random.randrange(3, 4)  # Columns for consistent alignment
    row_step = random.randrange(1, 2)  # Columns for consistent alignment
    character_positions = {x: [np.random.randint(0, height) for _ in range(height // row_step)] for x in range(0, width, column_step)}
    character_opacities = {x: [np.random.random() for _ in range(height // row_step)] for x in range(0, width, column_step)}
    character_values = {x: [np.random.choice(['0', '1']) for _ in range(height // row_step)] for x in range(0, width, column_step)}

def update_character_positions(height, gray, speedA, speedB, opacA):
    global character_positions, character_opacities, character_values
    updated_positions = {}
    updated_opacities = {}
    updated_values = {}
    for x, y_list in character_positions.items():
        new_y_list = []
        new_opacity_list = []
        new_value_list = []
        for i, y in enumerate(y_list):
            intensity = gray[y % height, x] if y < height else 0
            speed = int(intensity / 255 * -speedA) + speedB  # Speed based on pixel intensity

            new_y = y + speed
            new_opacity = (opacA / 100) + intensity / 255  # Opacity based on pixel intensity

            # Flip the character value randomly
            char = character_values[x][i]
            if np.random.rand() < 0.1:  # 10% chance to flip
                char = '1' if char == '0' else '0'

            new_y_list.append(new_y if new_y < height else 0)
            new_opacity_list.append(new_opacity)
            new_value_list.append(char)

        # Add new characters at the top of each column in every frame
        new_y_list.insert(random.randrange(int(height / 4), int(3 * height / 4)), random.randrange(int(height / 4), int(height)))
        new_opacity_list.insert(0, np.random.random())
        new_value_list.insert(0, np.random.choice(['1', '0']))

        updated_positions[x] = new_y_list
        updated_opacities[x] = new_opacity_list
        updated_values[x] = new_value_list

    return updated_positions, updated_opacities, updated_values

def apply_matrix_effect(frame, speedA, speedB, opacA, opacB, size):
    global character_positions, character_opacities, character_values

    # Ensure the frame has three channels
    if len(frame.shape) == 2 or frame.shape[2] == 1:  # Grayscale frame
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Auto-adjust contrast
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Map pixel intensity to green tint (Matrix style)
    green_frame = np.zeros_like(frame)
    green_frame[:, :, 1] = gray  # Assign grayscale values to the green channel

    # Update falling "1s and 0s" positions, opacities, and values
    height, width = gray.shape
    if not character_positions:
        initialize_character_positions(width, height)
    updated_positions, updated_opacities, updated_values = update_character_positions(height, gray, speedA, speedB, opacA)

    # Generate falling characters
    overlay = np.zeros_like(frame, dtype=np.uint8)
    for x, y_list in updated_positions.items():
        for i, y in enumerate(y_list):
            if 0 <= y < height:
                char = updated_values[x][i]
                opacity = int(255 * updated_opacities[x][i]) * opacB / 10  # Scale opacity to [0, 255]
                cv2.putText(
                    overlay, char, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    size, (opacity, opacity, opacity), 1, lineType=cv2.LINE_AA
                )
    character_positions.update(updated_positions)
    character_opacities.update(updated_opacities)
    character_values.update(updated_values)

    # Merge the overlay with the green frame
    mask = overlay[:, :, 1] > 0
    green_frame[mask] = overlay[mask]

    green_frame = cv2.normalize(overlay, None, 0, 255, cv2.NORM_MINMAX)
    return green_frame

def process_frames(frames_folder, green_frames_folder):
    os.makedirs(green_frames_folder, exist_ok=True)
    frames = sorted([f for f in os.listdir(frames_folder) if f.endswith('.png')])

    for frame_name in frames:
        frame_path = os.path.join(frames_folder, frame_name)
        frame = cv2.imread(frame_path)

        # Convert to green-tinted frames
        gray = cv2.normalize(frame, None, 255, 0, cv2.NORM_MINMAX)
        gray = cv2.normalize(gray, None, 255, 0, cv2.NORM_MINMAX)
        green_frame = np.zeros_like(frame)
        green_frame = cv2.merge([gray, gray, gray])

        # Save green-tinted frame
        green_frame_path = os.path.join(green_frames_folder, frame_name)
        cv2.imwrite(green_frame_path, green_frame)

    print(f"Green frames saved to {green_frames_folder}")

def create_chunked_gifs(green_frames_folder, output_gif_prefix):
    frames = sorted([f for f in os.listdir(green_frames_folder) if f.endswith('.png')])
    images = [Image.open(os.path.join(green_frames_folder, frame)) for frame in frames]

    num_frames = len(images)
    chunks = 512
    chunk_size = 8
    nb_chunks = 4

    for i in range(chunks):
        selected_frames = []
        first_chunk = None

        for chunk_idx in range(nb_chunks):
            if chunk_idx == 0:  # Generate the first chunk
                start_idx = random.randint(0, num_frames - chunk_size)
                first_chunk = images[start_idx: start_idx + chunk_size]
                selected_frames.extend(first_chunk)

            elif chunk_idx == nb_chunks - 1:  # Fourth chunk is the reverse of the first chunk
                if first_chunk is not None:
                    selected_frames.extend(first_chunk)

            else:  # Other chunks (2nd and 3rd)
                start_idx = random.randint(0, num_frames - chunk_size)
                selected_frames.extend(images[start_idx: start_idx + chunk_size])

        # Apply Matrix effect to each selected frame
        processed_frames = []
        prob = random.randrange(int(1), int(100))
        sign = 1
        if prob < 20:
            sign = -1

        valA = sign * random.randrange(int(50), int(60))
        valB = sign * random.randrange(int(8), int(25))
        size = random.randrange(1, 3) / 10
        speedA = valA
        speedB = valB
        if prob < 50:
            speedB = valA
            speedA = valB
        opacA = random.randrange(int(4), int(20))
        opacB = random.randrange(int(50), int(80))

        for frame in selected_frames:
            frame_array = np.array(frame)
            processed_frame = apply_matrix_effect(frame_array, speedA, speedB, opacA, opacB, size)
            processed_frames.append(Image.fromarray(processed_frame))

        output_gif = f"{output_gif_prefix}_{i + 1}_{speedA}_{speedB}_{opacA}_{opacB}.gif"

        # Calculate total playback time (in milliseconds) and adjust frame duration dynamically
        total_playback_time = 5000  # Total playback time for the GIF in milliseconds (e.g., 5 seconds)
        frame_count = len(processed_frames) - 10
        frame_duration = max(1, total_playback_time // frame_count)  # Ensure at least 1 ms per frame

        processed_frames[9].save(
            output_gif,
            save_all=True,
            append_images=processed_frames[10:],
            duration=80,  # Dynamically calculated frame duration
            loop=0
        )

        print(f"Chunked GIF saved as {output_gif}")