import torch
import torch.nn as nn
from models.common_fft_modules import RevIN

class LSTMForecaster(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        self.pred_len = config.pred_len
        self.num_layers = getattr(config, 'num_layers', 2)
        self.dropout = getattr(config, 'dropout', 0.2)
        self.use_revin = getattr(config, 'use_revin', True)

        self.revin = RevIN(num_features=self.input_size, eps=1e-5, affine=True)

        # Encode the history into recurrent hidden states.
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0
        )

        # Decode the final hidden state into the complete forecast horizon.
        self.fc = nn.Linear(self.hidden_size, self.pred_len * self.output_size)

    def forward(self, x, target_slice=None):
        """
        Forecast a complete horizon from the input history.
        x shape: (batch_size, seq_len, input_size)
        target_slice selects the target channels for RevIN denormalization.
        """
        # Normalize each input instance.
        if self.use_revin:
            x = self.revin(x, 'norm')

        # Initialize recurrent states with zeros.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # out shape: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Use the last state as the history representation.
        last_hidden_state = out[:, -1, :]  # shape: (batch_size, hidden_size)

        # Project once to every future step and target channel.
        # pred shape: (batch_size, pred_len * output_size)
        pred = self.fc(last_hidden_state)

        # Restore [batch, horizon, target] layout.
        pred = pred.view(-1, self.pred_len, self.output_size)

        # Return predictions to the original scale.
        if self.use_revin:
            pred = self.revin(pred, 'denorm', target_slice=target_slice)

        return pred
