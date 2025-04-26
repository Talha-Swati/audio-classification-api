import os
import librosa
import numpy as np
import tensorflow as tf
from supabase import create_client, Client
from typing import List, Dict, Any
from retrying import retry
from noisereduce import reduce_noise
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import JSONResponse
import tempfile
import uuid
from fastapi import FastAPI
app = FastAPI()
import os
if os.getenv("VERCEL"):
    app.root_path = "/api"

# Initialize FastAPI app
app = FastAPI(
    title="Audio Classification API",
    description="API for uploading audio files to Supabase and predicting labels using a TFLite model.",
    version="1.0.0"
)

# Load environment variables
load_dotenv()

# Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://eusytuqmwmorhzcchqfm.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImV1c3l0dXFtd21vcmh6Y2NocWZtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDE4OTAwMDEsImV4cCI6MjA1NzQ2NjAwMX0.P0a7ElL-WiWjAYmT0KxzohHNMiQyGe2dbVJOcxv7PYE")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load TFLite model
MODEL_PATH = os.getenv("MODEL_PATH", "audio_classification_model.tflite")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label encoder
LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH", "label_encoder.pkl")
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Feature extraction with normalization
def extract_features(y, sr):
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    features = np.concatenate((mfccs, spectral_contrast, mel_spectrogram, chroma))
    
    # Normalize features to [0, 1]
    features_min = np.min(features)
    features_max = np.max(features)
    if features_max != features_min:
        features = (features - features_min) / (features_max - features_min)
    return features

# Retry logic for Supabase
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def list_files(bucket: str, path: str) -> List[Dict[str, Any]]:
    files = supabase.storage.from_(bucket).list(path=path)
    if not isinstance(files, list):
        raise ValueError("Invalid Supabase list response")
    return files

# Prediction function (same as provided)
def predict_latest():
    temp_file = "temp.wav"

    try:
        # Fetch the latest audio file from Supabase
        files = list_files("user-recordings", "temp")
        if not files:
            raise ValueError("No audio files found in Supabase bucket")

        files = [f for f in files if 'name' in f and 'created_at' in f]
        files.sort(key=lambda x: x['created_at'], reverse=True)
        latest_file = files[0]
        path = f"temp/{latest_file['name']}"
        print(f"Fetching audio file: {path}")

        # Download from Supabase
        data = supabase.storage.from_("user-recordings").download(path)
        with open(temp_file, "wb") as f:
            f.write(data)

        # Load audio (already in WAV format, 16kHz)
        y, sr = librosa.load(temp_file, sr=16000)
        if len(y) == 0:
            raise ValueError("Empty or invalid audio")

        # Noise reduction
        y = reduce_noise(y, sr=sr)

        # Extract features
        features = extract_features(y, sr)
        if len(features) != 72:
            raise ValueError(f"Invalid features extracted, expected 72 but got {len(features)}")

        print(f"Extracted features (first 5): {features[:5]}")

        # Reshape features to match the expected 3D shape (1, 72, 1)
        input_data = np.expand_dims(features, axis=(0, -1)).astype(np.float32)
        print(f"Input shape after reshaping: {input_data.shape}")

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # Get the predicted label and confidence
        label_index = np.argmax(prediction[0])
        confidence = float(prediction[0][label_index])
        predicted_label = label_encoder.inverse_transform([label_index])[0]

        print(f"Prediction probabilities: {prediction[0]}")
        print(f"Predicted label: {predicted_label}, Confidence: {confidence}")

        return predicted_label, confidence

    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

# FastAPI endpoint to upload audio and predict
@app.post("/upload-and-predict")
async def upload_and_predict(file: UploadFile = File(...)):
    try:
        # Validate file type (ensure it's a WAV file)
        if not file.filename.endswith('.wav'):
            raise HTTPException(status_code=400, detail="Only WAV files are supported")

        # Read the uploaded file
        contents = await file.read()

        # Create a unique filename using UUID
        unique_filename = f"{uuid.uuid4()}.wav"
        supabase_path = f"temp/{unique_filename}"

        # Upload to Supabase Storage
        supabase.storage.from_("user-recordings").upload(
            path=supabase_path,
            file=contents,
            file_options={"content-type": "audio/wav"}
        )

        print(f"Uploaded file to Supabase: {supabase_path}")

        # Run prediction on the latest file
        label, confidence = predict_latest()

        # Return the prediction result
        return JSONResponse(content={
            "status": "success",
            "predicted_label": label,
            "confidence": confidence,
            "uploaded_file": supabase_path
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
