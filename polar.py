import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, gru_output):
        # gru_output: (batch_size, seq_len, hidden_size)
        attention_scores = self.attention_layer(gru_output)  # (batch_size, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        attention_weights = attention_weights.unsqueeze(-1)  # (batch_size, seq_len, 1)
        weighted_sum = torch.sum(attention_weights * gru_output, dim=1)  # (batch_size, hidden_size)
        return weighted_sum, attention_weights
# 모델 정의
class CNN_GRU_Model(nn.Module):
    def __init__(self, num_classes):
        super(CNN_GRU_Model, self).__init__()
        
        # Define layers for channel 1
        self.conv1_channel_11 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=3, padding=1)
        self.conv1_channel_12 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=5, padding=2)
        self.conv1_channel_13 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=7, padding=3)
        self.conv1_channel_1 = nn.Conv1d(in_channels=106, out_channels=64, kernel_size=7)

        self.conv2_channel_1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5)
        self.conv3_channel_1 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3)

        self.pool3_channel_1 = nn.MaxPool1d(kernel_size=2)

        self.compress1 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1)  # 1x1 Conv1d
        self.gru_channel_1 = nn.GRU(494, 494, batch_first=True, bidirectional=True)

        # Define layers for channel 2
        self.conv1_channel_21 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=3, padding=1)
        self.conv1_channel_22 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=5,padding=2)
        self.conv1_channel_23 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=7, padding=3)
        
        self.conv1_channel_2 = nn.Conv1d(in_channels=106, out_channels=64, kernel_size=7)
        
        self.conv2_channel_2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5)
        self.pool2_channel_2 = nn.MaxPool1d(kernel_size=2)
        self.conv3_channel_2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3)
        self.pool3_channel_2 = nn.MaxPool1d(kernel_size=2)
        self.compress2 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1)
        self.gru_channel_2 = nn.GRU(494, 494, batch_first=True, bidirectional=True)
        
        self.attention_1 = Attention(494*2)
        self.attention_2 = Attention(494*2)

        # Define the fully connected layer
        self.fc1 = nn.Linear(494*4, 256)  # Adjust this depending on the number of classes
        self.fc2 = nn.Linear(256, 8)  # Adjust this depending on the number of classes
        self.fc3 = nn.Linear(8, num_classes)  # Adjust this depending on the number of classes
    
    def forward(self, x):
        # 채널 분리 및 각 행을 독립적인 시퀀스로 취급
        
        x_channel_1 = x[:, 0, :, :].transpose(1, 2).reshape(x.size(0), x.size(2), -1)
        x_channel_2 = x[:, 1, :, :].transpose(1, 2).reshape(x.size(0), x.size(2), -1)
        
        # 첫 번째 채널에 대한 CNN
        
        x1 = self.conv1_channel_11(x_channel_1)
        x1 = self.conv1_channel_12(x1)
        x1 = self.conv1_channel_13(x1)
        
        
        x1 = F.relu(self.conv1_channel_1(x1))
        x1 = F.relu(self.conv2_channel_1(x1))
        x1 = self.pool3_channel_1(F.relu(self.conv3_channel_1(x1)))
        x1 = self.compress1(x1)
        
        
        x1 = x1.view(x1.size(0), -1, 494)  # Flatten
        
        # 두 번째 채널에 대한 CNN
        
        x2 = self.conv1_channel_21(x_channel_2)
        x2 = self.conv1_channel_22(x2)
        x2 = self.conv1_channel_23(x2)
        
        
        x2 = F.relu(self.conv1_channel_2(x2))
        x2 = F.relu(self.conv2_channel_2(x2))
        x2 = self.pool3_channel_2(F.relu(self.conv3_channel_2(x2)))
        x2 = self.compress2(x2)
        x2 = x2.view(x2.size(0), -1, 494)  # Flatten
        
        # GRU
        x1, _ = self.gru_channel_1(x1)
        x2, _ = self.gru_channel_2(x2)
        #x1 = x1[:, -1, :]  # 마지막 GRU 출력만 사용
        #x2 = x2[:, -1, :]  # 마지막 GRU 출력만 사용
        weighted_sum_1 , attention_weights_1 = self.attention_1(x1)
        weighted_sum_2 , attention_weights_2 = self.attention_1(x2)
        
        
        # GRU 출력 결합 및 FC 레이어
        x = torch.cat((weighted_sum_1, weighted_sum_2), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x

class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_classes):
        super(CNN_LSTM_Model, self).__init__()
        
        # Define layers for channel 1
        self.conv1_channel_11 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=3, padding=1)
        self.conv1_channel_12 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=5, padding=2)
        self.conv1_channel_13 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=7, padding=3)
        self.conv1_channel_1 = nn.Conv1d(in_channels=106, out_channels=64, kernel_size=7)

        self.conv2_channel_1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5)
        self.conv3_channel_1 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3)

        self.pool3_channel_1 = nn.MaxPool1d(kernel_size=2)

        self.compress1 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1)  # 1x1 Conv1d
        self.lstm_channel_1 = nn.LSTM(494, 494, batch_first=True, bidirectional=True)

        # Define layers for channel 2
        self.conv1_channel_21 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=3, padding=1)
        self.conv1_channel_22 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=5,padding=2)
        self.conv1_channel_23 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=7, padding=3)
        
        self.conv1_channel_2 = nn.Conv1d(in_channels=106, out_channels=64, kernel_size=7)
        
        self.conv2_channel_2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5)
        self.pool2_channel_2 = nn.MaxPool1d(kernel_size=2)
        self.conv3_channel_2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3)
        self.pool3_channel_2 = nn.MaxPool1d(kernel_size=2)
        self.compress2 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1)
        self.lstm_channel_2 = nn.LSTM(494, 494, batch_first=True, bidirectional=True)

        self.attention_1 = Attention(494*2)
        self.attention_2 = Attention(494*2)

        # Define the fully connected layer
        self.fc1 = nn.Linear(494*4, 256)  # Adjust this depending on the number of classes
        self.fc2 = nn.Linear(256, 8)  # Adjust this depending on the number of classes
        self.fc3 = nn.Linear(8, num_classes)  # Adjust this depending on the number of classes
    
    def forward(self, x):
        # 채널 분리 및 각 행을 독립적인 시퀀스로 취급
        
        x_channel_1 = x[:, 0, :, :].transpose(1, 2).reshape(x.size(0), x.size(2), -1)
        x_channel_2 = x[:, 1, :, :].transpose(1, 2).reshape(x.size(0), x.size(2), -1)
        
        # 첫 번째 채널에 대한 CNN
        
        x1 = self.conv1_channel_11(x_channel_1)
        x1 = self.conv1_channel_12(x1)
        x1 = self.conv1_channel_13(x1)
        
        
        x1 = F.relu(self.conv1_channel_1(x1))
        x1 = F.relu(self.conv2_channel_1(x1))
        x1 = self.pool3_channel_1(F.relu(self.conv3_channel_1(x1)))
        x1 = self.compress1(x1)
        
        
        x1 = x1.view(x1.size(0), -1, 494)  # Flatten
        
        # 두 번째 채널에 대한 CNN
        
        x2 = self.conv1_channel_21(x_channel_2)
        x2 = self.conv1_channel_22(x2)
        x2 = self.conv1_channel_23(x2)
        
        
        x2 = F.relu(self.conv1_channel_2(x2))
        x2 = F.relu(self.conv2_channel_2(x2))
        x2 = self.pool3_channel_2(F.relu(self.conv3_channel_2(x2)))
        x2 = self.compress2(x2)
        x2 = x2.view(x2.size(0), -1, 494)  # Flatten
        
        # GRU
        x1, _ = self.lstm_channel_1(x1)
        x2, _ = self.lstm_channel_2(x2)

        weighted_sum_1 , attention_weights_1 = self.attention_1(x1)
        weighted_sum_2 , attention_weights_2 = self.attention_1(x2)
        
        
        # GRU 출력 결합 및 FC 레이어
        x = torch.cat((weighted_sum_1, weighted_sum_2), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x
    


class CNN_TRANS_Model(nn.Module):
    def __init__(self, num_classes):
        super(CNN_TRANS_Model, self).__init__()
        
        # Define layers for channel 1
        self.conv1_channel_11 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=3, padding=1)
        self.conv1_channel_12 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=5, padding=2)
        self.conv1_channel_13 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=7, padding=3)

        self.conv1_channel_1 = nn.Conv1d(in_channels=106, out_channels=64, kernel_size=7)
        self.conv2_channel_1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5)
        self.conv3_channel_1 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3)

        self.pool3_channel_1 = nn.MaxPool1d(kernel_size=2)

        self.compress1 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1)  # 1x1 Conv1d
        self.encoder_layer_1 = nn.TransformerEncoderLayer(
            d_model=494,  # 입력 차원
            nhead=13,      # Multi-head attention에서의 head 수
            dim_feedforward=2048,  # Feedforward 네트워크의 차원
            dropout=0.1   # 드롭아웃 비율
        )
        self.transformer_encoder_1 = nn.TransformerEncoder(self.encoder_layer_1, num_layers=6)

        # Define layers for channel 2
        self.conv1_channel_21 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=3, padding=1)
        self.conv1_channel_22 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=5,padding=2)
        self.conv1_channel_23 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=7, padding=3)
        
        self.conv1_channel_2 = nn.Conv1d(in_channels=106, out_channels=64, kernel_size=7)
        
        self.conv2_channel_2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5)
        self.pool2_channel_2 = nn.MaxPool1d(kernel_size=2)
        self.conv3_channel_2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3)
        self.pool3_channel_2 = nn.MaxPool1d(kernel_size=2)
        self.compress2 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1)
        self.encoder_layer_2 = nn.TransformerEncoderLayer(
            d_model=494,  # 입력 차원
            nhead=13,      # Multi-head attention에서의 head 수
            dim_feedforward=2048,  # Feedforward 네트워크의 차원
            dropout=0.1   # 드롭아웃 비율
        )
        self.transformer_encoder_2 = nn.TransformerEncoder(self.encoder_layer_2, num_layers=6)

        self.attention_1 = Attention(494)
        self.attention_2 = Attention(494)

        # Define the fully connected layer
        self.fc1 = nn.Linear(494*2, 256)  # Adjust this depending on the number of classes
        self.fc2 = nn.Linear(256, 8)  # Adjust this depending on the number of classes
        self.fc3 = nn.Linear(8, num_classes)  # Adjust this depending on the number of classes
    
    def forward(self, x):
        # 채널 분리 및 각 행을 독립적인 시퀀스로 취급
        
        x_channel_1 = x[:, 0, :, :].transpose(1, 2).reshape(x.size(0), x.size(2), -1)
        x_channel_2 = x[:, 1, :, :].transpose(1, 2).reshape(x.size(0), x.size(2), -1)
        
        # 첫 번째 채널에 대한 CNN
        
        x1 = self.conv1_channel_11(x_channel_1)
        x1 = self.conv1_channel_12(x1)
        x1 = self.conv1_channel_13(x1)
        
        
        x1 = F.relu(self.conv1_channel_1(x1))
        x1 = F.relu(self.conv2_channel_1(x1))
        x1 = self.pool3_channel_1(F.relu(self.conv3_channel_1(x1)))
        x1 = self.compress1(x1)
        
        
        x1 = x1.view(x1.size(0), -1, 494)  # Flatten
        
        # 두 번째 채널에 대한 CNN
        
        x2 = self.conv1_channel_21(x_channel_2)
        x2 = self.conv1_channel_22(x2)
        x2 = self.conv1_channel_23(x2)
        
        
        x2 = F.relu(self.conv1_channel_2(x2))
        x2 = F.relu(self.conv2_channel_2(x2))
        x2 = self.pool3_channel_2(F.relu(self.conv3_channel_2(x2)))
        x2 = self.compress2(x2)
        x2 = x2.view(x2.size(0), -1, 494)  # Flatten
        
        # transformer
        x1 = self.transformer_encoder_1(x1)
        x2 = self.transformer_encoder_2(x2)

        weighted_sum_1 , attention_weights_1 = self.attention_1(x1)
        weighted_sum_2 , attention_weights_2 = self.attention_1(x2)
        
        
        # GRU 출력 결합 및 FC 레이어
        x = torch.cat((weighted_sum_1, weighted_sum_2), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x