import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms as transforms
from PIL import Image
import googlenetv4

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
class Polar_gnet_Model(nn.Module):
    def __init__(self, num_classes):
        super(Polar_gnet_Model, self).__init__()
        k = 192
        l = 224
        m = 256
        n = 384
        num_classes = 1000
        dropout_prob = 0.0
    
        self.features = nn.Sequential(
            googlenetv4.InceptionV4Stem(1),
            googlenetv4.InceptionA(384),
            googlenetv4.InceptionA(384),
            googlenetv4.InceptionA(384),
            googlenetv4.InceptionA(384),
            googlenetv4.ReductionA(384, k, l, m, n),
            googlenetv4.InceptionB(1024),
            googlenetv4.InceptionB(1024),
            googlenetv4.InceptionB(1024),
            googlenetv4.InceptionB(1024),
            googlenetv4.ReductionB(1024),
            googlenetv4.InceptionC(1536),
            googlenetv4.InceptionC(1536),
            googlenetv4.InceptionC(1536),
        )
        
        
        self.global_average_pooling_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.global_average_pooling_2 = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(1536*2, num_classes)

    def forward(self, x):
        # 채널 분리 및 각 행을 독립적인 시퀀스로 취급
        
        # 입력 텐서의 크기 확인 (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()
        
        # 첫 번째 채널 추출 및 unsqueeze로 채널 차원 추가 (batch_size, 1, height, width)
        ### 이거 정사각형 요구해서 안됨 어짜피
        x_channel_1 = x[:, 0:1, :, :]
        x_channel_2 = x[:, 1:2, :, :]
        
        x1 = self.features(x_channel_1)
        x2 = self.features(x_channel_2)
        
        x1 = self.global_average_pooling_1(x1)
        x2 = self.global_average_pooling_2(x2)
        
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        
        x = torch.cat((x1, x2), dim=1)
        
        x = self.linear(x)
        
        return x
        
