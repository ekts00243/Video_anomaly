# train_model.py
import os
import numpy as np
import tensorflow as tf
from preprocessing.video_loader import VideoLoader
from models.lstm_ae import LSTMAutoencoder
from config.config import Config

def prepare_training_data():
    """
    Prepare training data from videos in input directory
    Returns:
        np.ndarray: Preprocessed video frames for training
    """
    # Get list of video files
    video_files = [
        os.path.join(Config.VIDEO_INPUT_DIR, f)
        for f in os.listdir(Config.VIDEO_INPUT_DIR)
        if f.endswith(('.mp4', '.avi', '.mov'))
    ]
    # Collect frames from all videos
    all_frames = []
    for video_path in video_files:
        frames = VideoLoader.load_video(video_path)
        all_frames.append(frames)
    # Concatenate frames from all videos
    X = np.concatenate(all_frames)
    # Reshape for LSTM input
    X = X.reshape((-1, X.shape[1], np.prod(X.shape[2:])))
    return X
def train_anomaly_detector():
    """
    Train the anomaly detection model and save it
    """
    # Prepare training data
    X_train = prepare_training_data()
    # Initialize the LSTM Autoencoder
    autoencoder = LSTMAutoencoder(input_shape=X_train.shape[1:])
    print("Starting model training...")
    # Train the model
    autoencoder.train(X_train)
    # Save the trained model
    print(f"Saving model to {Config.MODEL_SAVE_PATH}")
    autoencoder.model.save(Config.MODEL_SAVE_PATH)
    print("Model training completed and saved successfully!")
def main():
    # Ensure input directory exists and has videos
    if not os.listdir(Config.VIDEO_INPUT_DIR):
        print(f"Error: No videos found in {Config.VIDEO_INPUT_DIR}")
        print("Please add training videos to this directory.")
        return
    # Train the model
    train_anomaly_detector()
if __name__ == "__main__":
    main()