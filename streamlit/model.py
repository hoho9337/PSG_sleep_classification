import torch
import torch.nn as nn
import torch.optim as optim

class CNNLSTMModel(nn.Module):
    def __init__(self, num_channels, num_filters, lstm_hidden_size, num_classes):
        super(CNNLSTMModel, self).__init__()
        self.num_filters = num_filters
        self.lstm_hidden_size = lstm_hidden_size
        self.dropout = nn.Dropout(p=0.5)

        # Define the CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=num_filters * 2, hidden_size=lstm_hidden_size, batch_first=True)

        # Define the fully connected layer
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        batch_size, num_epochs, num_channels, sequence_length = x.size()

        # Reshape input to (batch_size * num_epochs, num_channels, sequence_length)
        x = x.view(batch_size * num_epochs, num_channels, sequence_length)

        # Pass through CNN
        x = self.cnn(x)

        # Get the new sequence length after the convolution and pooling operations
        _, num_filters, reduced_sequence_length = x.size()

        # Reshape output to (batch_size, num_epochs, reduced_sequence_length, num_filters)
        x = x.view(batch_size, num_epochs, reduced_sequence_length, num_filters)

        # Permute to (batch_size, num_epochs, reduced_sequence_length, num_filters)
        x = x.permute(0, 1, 3, 2)

        # Reshape to (batch_size * num_epochs, reduced_sequence_length, num_filters)
        x = x.contiguous().view(batch_size * num_epochs, reduced_sequence_length, num_filters)

        # Pass through LSTM
        x, _ = self.lstm(x)

        # Reshape to (batch_size, num_epochs, reduced_sequence_length, lstm_hidden_size)
        x = x.view(batch_size, num_epochs, reduced_sequence_length, self.lstm_hidden_size)
        x = self.dropout(x)
        # Select the last time step for each epoch (many-to-one LSTM)
        x = x[:, :, -1, :]

        # Pass through fully connected layer
        x = self.fc(x)

        return x