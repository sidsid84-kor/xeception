import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms as transforms
from PIL import Image

def linear_polar_gpu(input_img, center):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_img = input_img.to(device)
    
    size = input_img.shape[2]  # width와 height가 같다고 가정
    max_radius = size/2  # 이미지 중심으로부터 구석까지의 최대 거리

    # 각도와 반경을 계산할 그리드 생성
    theta, r = torch.meshgrid(
        torch.linspace(0, 2 * np.pi, size, device=device),
        torch.linspace(0, max_radius, size, device=device),
        indexing='ij')
    x = r * torch.cos(theta) + center[0]
    y = r * torch.sin(theta) + center[1]

    # Normalize to -1 to 1 for grid_sample
    x_normalized = (x - size / 2) / (size / 2)
    y_normalized = (y - size / 2) / (size / 2)
    grid = torch.stack((x_normalized, y_normalized), dim=-1).unsqueeze(0)

    # grid_sample을 사용하여 극좌표 이미지 생성
    polar_img = F.grid_sample(input_img, grid, mode='bilinear', padding_mode='border', align_corners=False)
    polar_img = polar_img.rot90(1, [2, 3])

    return polar_img

def devide_img(img_tensor):
    # img_tensor는 [B, C, H, W] 형식의 텐서라고 가정 (배치 크기, 채널, 높이, 너비)
    # 이미지의 상단 부분 (0부터 160픽셀까지의 행)
    img_top = img_tensor[:, :, :160, :]
    
    # 이미지의 나머지 하단 부분 (160픽셀 이후부터 끝까지의 행)
    img_bottom = img_tensor[:, :, 160:, :]
    
    return img_top, img_bottom


def tensor_to_image(tensor, file_path):
    tensor = tensor.cpu().detach()  # GPU에서 계산된 텐서를 CPU로 이동
    tensor = tensor.squeeze(0)  # 첫 번째 차원이 배치라면 제거

    # 텐서 값 범위 확인 후 조정
    if tensor.max() <= 1.0:  # 텐서의 최대값이 1 이하라면 0-255 범위로 조정
        tensor = tensor * 255

    tensor = tensor.clamp(0, 255).byte()  # 값 범위를 0-255로 제한하고 uint8 형태로 변환
    image = Image.fromarray(tensor.numpy(), 'L')  # 'L'은 그레이스케일, 'RGB'는 컬러 이미지
    image.save(file_path)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers):
        super(CNNBlock, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()

        for i in range(num_layers):
            self.conv_layers.append(nn.Conv1d(in_channels if i == 0 else out_channels,
                                              out_channels, kernel_size, padding=kernel_size//2))
            self.bn_layers.append(nn.BatchNorm1d(out_channels))
            self.relu_layers.append(nn.ReLU())

    def forward(self, x):
        outputs = []
        for conv, bn, relu in zip(self.conv_layers, self.bn_layers, self.relu_layers):
            x = conv(x)
            outputs.append(x)  # Add convolution output
            x = bn(x)
            x = relu(x)
        return x, outputs  # Return the final output and list of convolution outputs

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
        
        
        self.channel_sizes = [128,64,32]
        self.repeat_num = 5
        self.lstm_hidden = 200 * self.repeat_num
        
        # Define layers for channel 1
        self.conv_block1_1 = CNNBlock(106, self.channel_sizes[0], 3, self.repeat_num)
        self.conv_block2_1 = CNNBlock(self.channel_sizes[0], self.channel_sizes[1], 5, self.repeat_num)
        self.conv_block3_1 = CNNBlock(self.channel_sizes[1], self.channel_sizes[2], 7, self.repeat_num)

        self.pool3_channel_1 = nn.MaxPool1d(kernel_size=2)

        self.compress1 = nn.Conv1d(in_channels= sum(self.channel_sizes) * self.repeat_num, out_channels=10*self.repeat_num, kernel_size=1)  # 1x1 Conv1d

        self.lstm_channel_1 = nn.LSTM(10*self.repeat_num, self.lstm_hidden, batch_first=True, bidirectional=True)

        # Define layers for channel 2
        # Define layers for channel 1
        self.conv_block1_2 = CNNBlock(106, self.channel_sizes[0], 3, self.repeat_num)
        self.conv_block2_2 = CNNBlock(self.channel_sizes[0], self.channel_sizes[1], 5, self.repeat_num)
        self.conv_block3_2 = CNNBlock(self.channel_sizes[1], self.channel_sizes[2], 7, self.repeat_num)
                
        self.pool3_channel_2 = nn.MaxPool1d(kernel_size=2)
        self.compress2 = nn.Conv1d(in_channels=sum(self.channel_sizes) * self.repeat_num, out_channels=10*self.repeat_num, kernel_size=1)
        
        self.lstm_channel_2 = nn.LSTM(10*self.repeat_num, self.lstm_hidden, batch_first=True, bidirectional=True)

        self.attention_1 = Attention(self.lstm_hidden*2)
        self.attention_2 = Attention(self.lstm_hidden*2)

        # Define the fully connected layer
        self.fc1 = nn.Linear(self.lstm_hidden*4, 1024)  # Adjust this depending on the number of classes
        self.fc2 = nn.Linear(1024, 512)  # Adjust this depending on the number of classes
        self.fc3 = nn.Linear(512, num_classes)  # Adjust this depending on the number of classes
    
    def forward(self, x):
        # 채널 분리 및 각 행을 독립적인 시퀀스로 취급
        
        #x_channel_1 = x[:, 0, :, :].transpose(1, 2).reshape(x.size(0), x.size(2), -1)
        #x_channel_2 = x[:, 1, :, :].transpose(1, 2).reshape(x.size(0), x.size(2), -1)
        
        
        x_channel_1 = x[:, 1, :, :]
        x_channel_2 = x[:, 2, :, :]
        
        # 예시 사용
        #tensor_to_image(x_channel_1[0], 'sample/output_channel_1.png')  # 첫 번째 샘플의 이미지 저장  
        #tensor_to_image(x_channel_2[0], 'sample/output_channel_2.png')
        
        #print("Original shape:", x.shape)
        #print("Transposed shape:", x_channel_1.shape)
        
        # 첫 번째 채널에 대한 CNN
        
        x1, outputs1_1 = self.conv_block1_1(x_channel_1)
        x1, outputs2_1 = self.conv_block2_1(x1)
        x1, outputs3_1 = self.conv_block3_1(x1)
        
        x1 = torch.cat(outputs1_1 + outputs2_1 + outputs3_1, dim=1)
        
        x1 = self.pool3_channel_1(x1)
        x1 = self.compress1(x1)
        
        
        #x1 = x1.view(x1.size(0), -1, 494)  # Flatten
        # 올바르게 차원을 조정하는 코드
        x1 = x1.permute(0, 2, 1)  # [batch_size, 10, 494]를 [batch_size, 494, 10]으로 변경

        
        # 두 번째 채널에 대한 CNN
        
        x2, outputs1_2 = self.conv_block1_2(x_channel_2)
        x2, outputs2_2 = self.conv_block2_2(x2)
        x2, outputs3_2 = self.conv_block3_2(x2)
        
        x2 = torch.cat(outputs1_2 + outputs2_2 + outputs3_2, dim=1)
        
        x2 = self.pool3_channel_2(x2)
        
        x2 = self.compress2(x2)
        x2 = x2.permute(0, 2, 1)
        #x2 = x2.view(x2.size(0), -1, 494)  # Flatten
        
        # GRU
        x1, _ = self.lstm_channel_1(x1)
        x2, _ = self.lstm_channel_2(x2)

        weighted_sum_1 , attention_weights_1 = self.attention_1(x1)
        weighted_sum_2 , attention_weights_2 = self.attention_2(x2)
        
        
        # GRU 출력 결합 및 FC 레이어
        x = torch.cat((weighted_sum_1, weighted_sum_2), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x
        
class CNN_LSTM_Model_original(nn.Module):
    def __init__(self, num_classes):
        super(CNN_LSTM_Model, self).__init__()
        
        self.lstm_hidden = 200
        
        # Define layers for channel 1
        self.conv1_channel_11 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=3, padding=1)
        self.conv1_channel_12 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=5, padding=2)
        self.conv1_channel_13 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=7, padding=3)
        self.bn1_channel_1 = nn.BatchNorm1d(106)
        self.bn1_channel_2 = nn.BatchNorm1d(106)
        self.bn1_channel_3 = nn.BatchNorm1d(106)

        self.conv1_channel_1 = nn.Conv1d(in_channels=106, out_channels=64, kernel_size=7)
        self.conv2_channel_1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5)
        self.conv3_channel_1 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3)

        self.pool3_channel_1 = nn.MaxPool1d(kernel_size=2)

        self.compress1 = nn.Conv1d(in_channels=16, out_channels=10, kernel_size=1)  # 1x1 Conv1d

        self.lstm_channel_1 = nn.LSTM(10, self.lstm_hidden, batch_first=True, bidirectional=True)

        # Define layers for channel 2
        self.conv1_channel_21 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=3, padding=1)
        self.conv1_channel_22 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=5,padding=2)
        self.conv1_channel_23 = nn.Conv1d(in_channels=106, out_channels=106, kernel_size=7, padding=3)
        
        self.bn2_channel_1 = nn.BatchNorm1d(106)
        self.bn2_channel_2 = nn.BatchNorm1d(106)
        self.bn2_channel_3 = nn.BatchNorm1d(106)
        
        self.conv1_channel_2 = nn.Conv1d(in_channels=106, out_channels=64, kernel_size=7)
        
        self.conv2_channel_2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5)
        self.pool2_channel_2 = nn.MaxPool1d(kernel_size=2)
        self.conv3_channel_2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3)
        self.pool3_channel_2 = nn.MaxPool1d(kernel_size=2)
        self.compress2 = nn.Conv1d(in_channels=16, out_channels=10, kernel_size=1)
        
        self.lstm_channel_2 = nn.LSTM(10, self.lstm_hidden, batch_first=True, bidirectional=True)

        self.attention_1 = Attention(self.lstm_hidden*2)
        self.attention_2 = Attention(self.lstm_hidden*2)

        # Define the fully connected layer
        self.fc1 = nn.Linear(self.lstm_hidden*4, 512)  # Adjust this depending on the number of classes
        self.fc2 = nn.Linear(512, 64)  # Adjust this depending on the number of classes
        self.fc3 = nn.Linear(64, num_classes)  # Adjust this depending on the number of classes
    
    def forward(self, x):
        # 채널 분리 및 각 행을 독립적인 시퀀스로 취급
        
        x_channel_1 = x[:, 0, :, :].transpose(1, 2).reshape(x.size(0), x.size(2), -1)
        x_channel_2 = x[:, 1, :, :].transpose(1, 2).reshape(x.size(0), x.size(2), -1)
        
        # 첫 번째 채널에 대한 CNN
        
        x1 = self.conv1_channel_11(x_channel_1)
        x1 = F.relu(self.bn1_channel_1(x1))
        x1 = self.conv1_channel_12(x1)
        x1 = F.relu(self.bn1_channel_2(x1))
        x1 = self.conv1_channel_13(x1)
        x1 = F.relu(self.bn1_channel_3(x1))
        
        x1 = F.relu(self.conv1_channel_1(x1))
        x1 = F.relu(self.conv2_channel_1(x1))
        x1 = self.pool3_channel_1(F.relu(self.conv3_channel_1(x1)))
        x1 = self.compress1(x1)
        
        
        #x1 = x1.view(x1.size(0), -1, 494)  # Flatten
        # 올바르게 차원을 조정하는 코드
        x1 = x1.permute(0, 2, 1)  # [batch_size, 10, 494]를 [batch_size, 494, 10]으로 변경

        
        # 두 번째 채널에 대한 CNN
        
        x2 = self.conv1_channel_21(x_channel_2)
        x2 = F.relu(self.bn2_channel_1(x2))
        x2 = self.conv1_channel_22(x2)
        x2 = F.relu(self.bn2_channel_2(x2))
        x2 = self.conv1_channel_23(x2)
        x2 = F.relu(self.bn2_channel_3(x2))
        
        
        x2 = F.relu(self.conv1_channel_2(x2))
        x2 = F.relu(self.conv2_channel_2(x2))
        x2 = self.pool3_channel_2(F.relu(self.conv3_channel_2(x2)))
        x2 = self.compress2(x2)
        x2 = x2.permute(0, 2, 1)
        #x2 = x2.view(x2.size(0), -1, 494)  # Flatten
        
        # GRU
        x1, _ = self.lstm_channel_1(x1)
        x2, _ = self.lstm_channel_2(x2)

        weighted_sum_1 , attention_weights_1 = self.attention_1(x1)
        weighted_sum_2 , attention_weights_2 = self.attention_2(x2)
        
        
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