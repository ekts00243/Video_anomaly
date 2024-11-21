# models/lstm_ae.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector
from tensorflow.keras.optimizers import Adam
from config.config import Config
class LSTMAutoencoder:
    def __init__(self, input_shape):
        """
        Initialize LSTM Autoencoder for Anomaly Detection
        Args:
            input_shape (tuple): Shape of input video frames
        """
        self.input_shape = input_shape
        self.model = self._build_model()
    def _build_model(self) -> Model:
        """
        Construct LSTM Autoencoder architecture
        Returns:
            Model: Compiled Keras model
        """
        # Encoder
        inputs = Input(shape=self.input_shape)
        encoded = LSTM(Config.LATENT_DIM, return_sequences=False)(inputs)
        # Decoder
        repeated = RepeatVector(self.input_shape[0])(encoded)
        decoded = LSTM(self.input_shape[1], return_sequences=True)(repeated)
        # Create autoencoder model
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=Config.LEARNING_RATE),
                            loss='mse')
        return autoencoder
    def train(self, X_train, epochs=None, batch_size=None):
        """
        Train the autoencoder model
        Args:
            X_train (np.ndarray): Training data
            epochs (int, optional): Number of training epochs
            batch_size (int, optional): Training batch size
        """
        epochs = epochs or Config.EPOCHS
        batch_size = batch_size or Config.BATCH_SIZE
        self.model.fit(X_train, X_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=True)
    def detect_anomalies(self, X_test, threshold=None):
        """
        Detect anomalies based on reconstruction error
        Args:
            X_test (np.ndarray): Test video data
            threshold (float, optional): Anomaly threshold
        Returns:
            np.ndarray: Anomaly labels
        """
        threshold = threshold or Config.RECONSTRUCTION_ERROR_THRESHOLD
        # Reconstruct input
        reconstructed = self.model.predict(X_test)
        # Calculate reconstruction error
        mse = np.mean(np.power(X_test - reconstructed, 2), axis=(1,2))
        # Label anomalies
        anomalies = mse > threshold
        return anomalies
