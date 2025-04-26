# Audio Classification API

A FastAPI application for audio classification using a TFLite model and Supabase for storage. This project allows users to upload WAV audio files, processes them for feature extraction, and predicts labels using a pre-trained model.

## Features
- Upload audio files to Supabase Storage.
- Extract audio features using Librosa (MFCCs, spectral contrast, mel spectrogram, chroma).
- Noise reduction with `noisereduce`.
- Predict labels using a TFLite model.
- API endpoints: `/upload-and-predict` and `/health`.

## Tech Stack
- **Backend**: FastAPI, Python
- **Storage**: Supabase
- **Audio Processing**: Librosa, noisereduce
- **Machine Learning**: TensorFlow Lite (TFLite)
- **Dependencies**: See `requirements.txt`

## Project Structure
