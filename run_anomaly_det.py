# detect_anomalies.py
import os
import sys
import numpy as np
import tensorflow as tf
from preprocessing.video_loader import VideoLoader
from config.config import Config
import matplotlib.pyplot as plt
def load_trained_model():
    """
    Load the pre-trained anomaly detection model
    Returns:
        tf.keras.Model: Loaded model
    """
    try:
        model = tf.keras.models.load_model(Config.MODEL_SAVE_PATH)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
def detect_anomalies_in_video(video_path, model):
    """
    Detect anomalies in a given video
    Args:
        video_path (str): Path to input video
        model (tf.keras.Model): Trained anomaly detection model
    Returns:
        tuple: Anomaly results and visualization
    """
    # Load and preprocess video
    frames = VideoLoader.load_video(video_path)
    X = frames.reshape((-1, frames.shape[1], np.prod(frames.shape[2:])))
    # Reconstruct input
    reconstructed = model.predict(X)
    # Calculate reconstruction error
    mse = np.mean(np.power(X - reconstructed, 2), axis=(1,2))
    # Apply threshold for anomaly detection
    anomalies = mse > Config.RECONSTRUCTION_ERROR_THRESHOLD
    # Visualization
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(mse, label='Reconstruction Error')
    plt.title('Reconstruction Error')
    plt.xlabel('Frame')
    plt.ylabel('Error')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(anomalies, label='Anomalies')
    plt.title('Anomaly Detection')
    plt.xlabel('Frame')
    plt.ylabel('Anomaly')
    plt.legend()
    plt.tight_layout()
    # Save results
    result_path = os.path.join(Config.OUTPUT_DIR,
                                f'anomaly_results_{os.path.basename(video_path)}.png')
    plt.savefig(result_path)
    # Print summary
    print(f"\nAnomaly Detection Results for {os.path.basename(video_path)}:")
    print(f"Total Frames: {len(X)}")
    print(f"Anomalous Frames: {np.sum(anomalies)}")
    print(f"Anomaly Percentage: {np.mean(anomalies)*100:.2f}%")
    print(f"Results visualization saved to: {result_path}")
    return anomalies, plt
def main():
    # Check if video path is provided
    if len(sys.argv) < 2:
        print("Usage: python detect_anomalies.py /path/to/your/test_video.mp4")
        sys.exit(1)
    # Load trained model
    model = load_trained_model()
    # Get video path from command line
    video_path = sys.argv[1]
    # Validate video path
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)
    # Detect anomalies
    detect_anomalies_in_video(video_path, model)
if __name__ == "__main__":
    main()