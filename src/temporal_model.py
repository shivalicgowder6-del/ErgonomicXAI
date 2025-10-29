import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ErgonomicTemporalModel:
    """
    A simple LSTM-based model to predict ergonomic risk over a time sequence.
    This demonstrates the concept of temporal modeling.
    """
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        """
        Initializes the LSTM model architecture.

        Args:
            input_size (int): The number of features in the input.
            hidden_layer_size (int): The number of neurons in the hidden LSTM layer.
            output_size (int): The number of output values.
        """
        self.model = self.LSTMModel(input_size, hidden_layer_size, output_size)
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.is_trained = False
        self.dummy_data = None # To store dummy data for the explainer

    class LSTMModel(nn.Module):
        """Inner class defining the PyTorch LSTM model."""
        def __init__(self, input_size, hidden_layer_size, output_size):
            super().__init__()
            self.hidden_layer_size = hidden_layer_size
            self.lstm = nn.LSTM(input_size, hidden_layer_size)
            self.linear = nn.Linear(hidden_layer_size, output_size)
            self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                                torch.zeros(1,1,self.hidden_layer_size))

        def forward(self, input_seq):
            lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
            predictions = self.linear(lstm_out.view(len(input_seq), -1))
            return predictions[-1]

    def train_with_dummy_data(self, n_samples=100, seq_length=8, epochs=5):
        """
        Trains the model on randomly generated dummy data for demonstration.
        In a real application, this would use the `train_temporal_model` function
        with actual risk score data.
        
        Args:
            n_samples (int): Number of dummy sequences to generate.
            seq_length (int): Length of each sequence.
            epochs (int): Number of training epochs.
        """
        print("Training temporal model with dummy data for prototype demonstration...")
        # Generate dummy data: sequences of increasing numbers
        X_train = []
        y_train = []
        for i in range(n_samples):
            start = np.random.rand() * 10
            seq = np.linspace(start, start + seq_length * 0.5, seq_length)
            X_train.append(seq[:-1])
            y_train.append(seq[-1])
            
        X_train = torch.FloatTensor(X_train).unsqueeze(-1)
        y_train = torch.FloatTensor(y_train).unsqueeze(-1)
        
        self.dummy_data = X_train # Save for SHAP explainer background

        for i in range(epochs):
            for seq, labels in zip(X_train, y_train):
                self.optimizer.zero_grad()
                self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size),
                                          torch.zeros(1, 1, self.model.hidden_layer_size))

                y_pred = self.model(seq)

                single_loss = self.loss_function(y_pred, labels)
                single_loss.backward()
                self.optimizer.step()

        self.is_trained = True
        print("Dummy training complete.")

    def predict(self, keypoints_flat_sequence):
        """
        Predicts the future risk score based on a sequence of keypoints.
        For the demo, this simplifies to using a pre-calculated REBA score.
        
        Args:
            keypoints_flat_sequence (np.ndarray): A flattened array of keypoints.
                                                   In a real scenario, this would be a sequence.

        Returns:
            np.ndarray: The predicted risk score.
        """
        if not self.is_trained:
            # Fallback for safety
            return np.array([[5.0]]) # Return a neutral middle score if not trained
        
        # In this prototype, we simulate a sequence from a single frame for simplicity.
        # A real implementation would buffer the last N frames.
        # We use a dummy sequence based on the input to feed the model.
        dummy_input_val = np.mean(keypoints_flat_sequence) / 10.0 # Normalize roughly
        test_seq = np.linspace(dummy_input_val, dummy_input_val + 7, 7)
        test_seq_tensor = torch.FloatTensor(test_seq).view(-1, 1)

        self.model.eval()
        with torch.no_grad():
            self.model.hidden = (torch.zeros(1, 1, self.model.hidden_layer_size),
                                 torch.zeros(1, 1, self.model.hidden_layer_size))
            prediction = self.model(test_seq_tensor)
            return prediction.numpy().reshape(1,1)

# Example usage (for testing this script directly)
if __name__ == '__main__':
    model = ErgonomicTemporalModel()
    model.train_with_dummy_data()
    
    # Create a dummy keypoint vector (33 landmarks * 3 coords = 99 features)
    dummy_keypoints = np.random.rand(99) 
    
    prediction = model.predict(dummy_keypoints)
    print(f"\nTest Prediction with dummy keypoints: {prediction[0][0]:.2f}")

