import os

class Config:
    MODEL_SAVE_PATH = './data/trained_models/anomaly_detector.h5'
    # Video Processing Configuration
    VIDEO_INPUT_DIR = './data/input_videos'
    FRAME_OUTPUT_DIR = './data/processed_frames'
    # Model Configurations
    MODEL_TYPE = 'lstm_autoencoder'
    INPUT_SHAPE = (640, 480, 3)  # Typical input size for deep learning models
    LATENT_DIM = 128
    # Training Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    # Anomaly Detection Thresholds
    RECONSTRUCTION_ERROR_THRESHOLD = 0.75
    # Logging and Output
    LOG_DIR = './logs'
    OUTPUT_DIR = './results'
    @classmethod
    def create_directories(cls):
        """Create necessary directories for the project."""
        os.makedirs(cls.VIDEO_INPUT_DIR, exist_ok=True)
        os.makedirs(cls.FRAME_OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
# Initialize directories when config is imported
Config.create_directories()