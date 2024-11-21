# main.py
import numpy as np
from preprocessing.video_loader import VideoLoader
from models.lstm_ae import LSTMAutoencoder
from config.config import Config
import os
def main():
    # Load and preprocess videos
    video_files = [os.path.join(Config.VIDEO_INPUT_DIR, f)
                   for f in os.listdir(Config.VIDEO_INPUT_DIR)
                   if f.endswith(('.mp4', '.avi', '.mov'))]
    # Process each video
    for video_path in video_files:
        # Load video frames
        frames = VideoLoader.load_video(video_path)
        # Reshape frames for LSTM input
        X = frames.reshape((-1, frames.shape[1], np.prod(frames.shape[2:])))
        # Initialize and train autoencoder
        autoencoder = LSTMAutoencoder(input_shape=X.shape[1:])
        # Split into train and test
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        # Train model
        autoencoder.train(X_train)
        # Detect anomalies
        anomalies = autoencoder.detect_anomalies(X_test)
        # Visualize or log anomalies
        print(f"Anomalies detected in {video_path}: {np.sum(anomalies)} / {len(anomalies)}")
if __name__ == "__main__":
    main()