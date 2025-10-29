"""
Fallback temporal model without PyTorch dependency
"""
import numpy as np

class ErgonomicTemporalModel:
    """
    Fallback temporal model that works without PyTorch
    """
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        """Initialize the fallback model"""
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.is_trained = False
        self.dummy_data = None

    def train_with_dummy_data(self, n_samples=100, seq_length=8, epochs=5):
        """
        Train the model on dummy data (simulated training)
        """
        print("Training temporal model with dummy data for prototype demonstration...")
        
        # Generate dummy training data
        X_train = []
        y_train = []
        for i in range(n_samples):
            start = np.random.rand() * 10
            seq = np.linspace(start, start + seq_length * 0.5, seq_length)
            X_train.append(seq[:-1])
            y_train.append(seq[-1])
        
        self.dummy_data = np.array(X_train)
        self.is_trained = True
        print("Dummy training complete.")

    def predict(self, keypoints_flat_sequence):
        """
        Predict future risk score based on keypoints
        """
        if not self.is_trained:
            # Return a neutral middle score if not trained
            return np.array([[5.0]])
        
        # Simple prediction based on keypoint characteristics
        # This simulates what the LSTM would do
        if len(keypoints_flat_sequence) == 0:
            return np.array([[5.0]])
        
        # Analyze keypoint characteristics
        mean_value = np.mean(keypoints_flat_sequence)
        std_value = np.std(keypoints_flat_sequence)
        
        # Generate prediction based on keypoint analysis
        # More variation in keypoints = higher risk prediction
        base_prediction = 5.0 + (std_value * 2.0) + (mean_value * 0.1)
        
        # Add some temporal variation
        temporal_factor = np.sin(mean_value * 0.1) * 2.0
        prediction = base_prediction + temporal_factor
        
        # Ensure prediction is within reasonable range
        prediction = max(1.0, min(12.0, prediction))
        
        return np.array([[prediction]])

# Example usage (for testing this script directly)
if __name__ == '__main__':
    model = ErgonomicTemporalModel()
    model.train_with_dummy_data()
    
    # Create a dummy keypoint vector (33 landmarks * 3 coords = 99 features)
    dummy_keypoints = np.random.rand(99) 
    
    prediction = model.predict(dummy_keypoints)
    print(f"\nTest Prediction with dummy keypoints: {prediction[0][0]:.2f}")
