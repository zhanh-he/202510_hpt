import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from feature_extractor import get_feature_extractor_and_bins
from pytorch_utils import move_data_to_device
from einops import rearrange

class Rearrange(nn.Module):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern
    def forward(self, x):
        return rearrange(x, self.pattern)

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            # nn.init.zeros_(layer.bias)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)
    # nn.init.zeros_(layer.bias)
    # nn.init.ones_(bn.weight)

def init_bilstm(lstm):
    """Initialize weights for a Bidirectional LSTM layer."""
    for name, param in lstm.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
            # nn.init.constant_(param, 0.0)

def init_gru(rnn):
    """Initialize GRU weights and biases for better convergence."""
    def _concat_init(tensor, inits):
        fan_in = tensor.shape[0] // len(inits)
        for i, fn in enumerate(inits):
            fn(tensor[i * fan_in:(i + 1) * fan_in])
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        bound = math.sqrt(3 / fan_in)
        nn.init.uniform_(tensor, -bound, bound)

    for i in range(rnn.num_layers):
        _concat_init(getattr(rnn, f'weight_ih_l{i}'),
                     [_inner_uniform, _inner_uniform, _inner_uniform])
        nn.init.constant_(getattr(rnn, f'bias_ih_l{i}'), 0.)

        _concat_init(getattr(rnn, f'weight_hh_l{i}'),
                     [_inner_uniform, _inner_uniform, nn.init.orthogonal_])
        nn.init.constant_(getattr(rnn, f'bias_hh_l{i}'), 0.)

class HPTConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum)
        self.init_weight()
    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
    def forward(self, input):
        """input: (batch_size, in_channels,  time_steps, freq_bins)
          output: (batch_size, out_channels, classes_num)"""
        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(1,2))
        return x


def _align_time_dim_tensors(*tensors):
    """Trim tensors along time dimension (dim=1) to the minimum shared length."""
    valid = [t for t in tensors if t is not None]
    if not valid:
        return tensors
    min_steps = min(t.size(1) for t in valid)
    aligned = []
    for t in tensors:
        if t is None:
            aligned.append(None)
        elif t.size(1) == min_steps:
            aligned.append(t)
        else:
            aligned.append(t[:, :min_steps])
    return tuple(aligned)

class OriginalModelHPT2020(nn.Module):
    def __init__(self, classes_num, input_shape, momentum):
        super().__init__()
        self.conv_block1 = HPTConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = HPTConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = HPTConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = HPTConvBlock(in_channels=96, out_channels=128, momentum=momentum)
        # Auto Calculate midfeat
        with torch.no_grad():
            dummy = torch.zeros((1, 1, 1000, input_shape))  # 1000个帧，freq维为 input_shape
            x = self.conv_block1(dummy)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            x = self.conv_block4(x)
            # Flatten later uses x.transpose(1,2).flatten(2) -> channels × freq
            midfeat = x.shape[1] * x.shape[3]
        self.fc5 = nn.Linear(in_features=midfeat, out_features=768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)
        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2,
                          bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.fc =  nn.Linear(in_features=512, out_features=classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_gru(self.gru)
        init_layer(self.fc)
    def forward(self, input):
        """Args: input: (batch_size, channels_num, time_steps, freq_bins)
        Outputs: output: (batch_size, time_steps, classes_num)
        Selections: #print("Shape after conv_block2:", x.shape)"""								  
        x = self.conv_block1(input)		# conB1 batch=8, chn=48,timstep=1001, mel=114 
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x)	  		# conB2 batch=8, chn=64,timstep=1001, mel=57 
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x)	  		# conB3 batch=8, chn=96,timstep=1001, mel=28 
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x)	  		# conB4 batch=8,chn=128,timstep=1001, mel=14 
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.transpose(1, 2).flatten(2)	# tranpose 8,1001,48,114 ; flat 8,1001,1792 
        x = F.relu(self.bn5(self.fc5(x).transpose(1,2)).transpose(1,2)) # batch=8, timstep=1001, class=768
        x = F.dropout(x, p=0.5, training=self.training)
        (x, _) = self.gru(x)			# gru batch=8, timstep=1001, class=512 
        x = F.dropout(x, p=0.5, training=self.training)
        output = torch.sigmoid(self.fc(x))	# out batch=8, timstep=1001, class=88 
        return output


class ModifiedModelHPT2020(nn.Module):
    def __init__(self, classes_num, midfeat, momentum):
        super().__init__()
        self.conv_block1 = HPTConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = HPTConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = HPTConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = HPTConvBlock(in_channels=96, out_channels=128, momentum=momentum)
        self.fc5 = nn.Linear(midfeat, 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)
        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2, # Cannot mute this, I forgot during the training
                          bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=1, bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_gru(self.gru) # Cannot mute this, I forgot during the training
        init_layer(self.fc)
        init_bilstm(self.bilstm)
    def forward(self, input):
        """Args: input: (batch_size, channels_num, time_steps, freq_bins)
        Outputs: output: (batch_size, time_steps, classes_num)"""
        x = self.conv_block1(input)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.2, training=self.training) # 0.5
        (x, _) = self.bilstm(x) # replace GRU with BiLSTM
        x = F.dropout(x, p=0.2, training=self.training) # 0.5
        output = torch.sigmoid(self.fc(x))
        return output


class Single_Velocity_HPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        sample_rate         = cfg.feature.sample_rate
        fft_size            = cfg.feature.fft_size
        frames_per_second   = cfg.feature.frames_per_second
        audio_feature       = cfg.feature.audio_feature
        classes_num         = cfg.feature.classes_num
        momentum            = 0.01
        self.feature_extractor, self.FRE = get_feature_extractor_and_bins(audio_feature, sample_rate, fft_size, frames_per_second)
        # midfeat = 1792
        self.bn0 = nn.BatchNorm2d(self.FRE, momentum)
        self.velocity_model = OriginalModelHPT2020(classes_num, self.FRE , momentum)  # OriginalModelHPT2020
        # self.velocity_model = OriginalModelHPT2020(classes_num, midfeat, momentum)  # OriginalModelHPT2020
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
    def forward(self, input):
        """
        Args: input: (batch_size, data_length)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}
        """
        x = self.feature_extractor(input)    	# batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x = x.unsqueeze(3)                      # batch=12, melbins=229, timsteps=1001, ch=1
        x = self.bn0(x)					        # batch=12, melbins=229, timsteps=1001, ch=1
        x = x.transpose(1, 3)			    	# batch=12, ch=1, timsteps=1001, melbins=229
        est_velocity = self.velocity_model(x)  	# batch=12, timsteps=1001, classes_num=88
        output_dict = {'velocity_output': est_velocity}
        return output_dict

class Dual_Velocity_HPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        sample_rate         = cfg.feature.sample_rate
        fft_size            = cfg.feature.fft_size
        frames_per_second   = cfg.feature.frames_per_second
        audio_feature       = cfg.feature.audio_feature
        classes_num         = cfg.feature.classes_num
        momentum            = 0.01
        self.feature_extractor, self.FRE = get_feature_extractor_and_bins(audio_feature, sample_rate, fft_size, frames_per_second)
        self.bn0 = nn.BatchNorm2d(self.FRE, momentum)
        self.velocity_model = OriginalModelHPT2020(classes_num, self.FRE, momentum)
        self.bilstm = nn.LSTM(input_size=88*2, hidden_size=256, num_layers=1, bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.velo_fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
        init_bilstm(self.bilstm)
        init_layer(self.velo_fc)
    def forward(self, input1, input2):
        """
        Args:
            input1: (batch_size, data_length) — 音频波形
            input2: (batch_size, time_steps, classes_num) — 辅助输入 (e.g., onset)
        Returns:
            {'velocity_output': (batch_size, time_steps, classes_num)}
        ---- 可选三种融合方式 ----
        TypeA: 直接拼接
            x = torch.cat((pre_velocity, input2), dim=2)
        TypeB: 带平方根加权（轻度依赖 pre_velocity）
            x = torch.cat((pre_velocity, (pre_velocity.detach() ** 0.5) * input2), dim=2)
        TypeC: 混合加权（适中依赖 pre_velocity + input2）
            x = torch.cat((pre_velocity, ((pre_velocity.detach() + input2) ** 0.5) * input2), dim=2)
        """
        # ====== 特征提取 ======
        x_feat = self.feature_extractor(input1)       # (B, Freq, T)
        x_feat = x_feat.unsqueeze(3)                  # (B, Freq, T, 1)
        x_feat = self.bn0(x_feat)
        x_feat = x_feat.transpose(1, 3)               # (B, 1, T, Freq)
        # ====== 预测初始 velocity ======
        pre_velocity = self.velocity_model(x_feat)    # (B, T, 88)
        pre_velocity, input2 = _align_time_dim_tensors(pre_velocity, input2)
        # ====== 融合策略 (选其一) ======
        # --- TypeA ---
        x = torch.cat((pre_velocity, input2), dim=2)
        # --- TypeB ---
        # x = torch.cat((pre_velocity, (pre_velocity.detach() ** 0.5) * input2), dim=2)
        # --- TypeC ---
        # x = torch.cat((pre_velocity, ((pre_velocity.detach() + input2) ** 0.5) * input2), dim=2)
        # ====== 双向LSTM + 输出层 ======
        (x, _) = self.bilstm(x)
        # x = F.dropout(x, p=0.5, training=self.training)  # optional
        upd_velocity = torch.sigmoid(self.velo_fc(x))
        return {'velocity_output': upd_velocity}


class Triple_Velocity_HPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        sample_rate         = cfg.feature.sample_rate           # 16000
        fft_size            = cfg.feature.fft_size              # 2048
        frames_per_second   = cfg.feature.frames_per_second     # 100
        audio_feature       = cfg.feature.audio_feature
        classes_num         = cfg.feature.classes_num
        momentum            = 0.01
        self.feature_extractor, self.FRE = get_feature_extractor_and_bins(audio_feature, sample_rate, fft_size, frames_per_second)
        self.bn0 = nn.BatchNorm2d(self.FRE, momentum)
        self.velocity_model = OriginalModelHPT2020(classes_num, self.FRE, momentum)
        self.bilstm = nn.LSTM(input_size=88 * 3, hidden_size=256, num_layers=1, bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.velo_fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
        init_bilstm(self.bilstm)
        init_layer(self.velo_fc)
    def forward(self, input1, input2, input3):
        """
        Args:
            input1: (batch_size, data_length) — 音频波形
            input2: (batch_size, time_steps, classes_num) — 辅助输入1 (e.g., onset)
            input3: (batch_size, time_steps, classes_num) — 辅助输入2 (e.g., frame)
        Returns:
            {'velocity_output': (batch_size, time_steps, classes_num)}
        ---- 可选三种融合方式 ----
        TypeA: 直接拼接
            x = torch.cat((pre_velocity, input2, input3), dim=2)
        TypeB: 带平方根加权（轻度依赖 pre_velocity）
            x = torch.cat((pre_velocity, (pre_velocity.detach() ** 0.5) * input2, input3), dim=2)
        TypeC: 混合加权（适中依赖 pre_velocity + input2）
            x = torch.cat((pre_velocity, ((pre_velocity.detach() + input2) ** 0.5) * input2, input3), dim=2)
        """
        # ===== 特征提取 =====
        x_feat = self.feature_extractor(input1)   # batch, FRE, T
        x_feat = x_feat.unsqueeze(3)              # batch, FRE, T, 1
        x_feat = self.bn0(x_feat)
        x_feat = x_feat.transpose(1, 3)           # batch, 1, T, FRE
        # ===== 预测初始 velocity =====
        pre_velocity = self.velocity_model(x_feat)  # batch, T, 88
        pre_velocity, input2, input3 = _align_time_dim_tensors(pre_velocity, input2, input3)
        # ===== 融合策略 (选其一) =====
        # --- TypeA ---
        x = torch.cat((pre_velocity, input2, input3), dim=2) 
        # --- TypeB ---
        # x = torch.cat((pre_velocity, (pre_velocity.detach() ** 0.5) * input2, input3), dim=2)
        # --- TypeC ---
        # x = torch.cat((pre_velocity, ((pre_velocity.detach() + input2) ** 0.5) * input2, input3), dim=2)
        # ===== 后续LSTM + FC =====
        (x, _) = self.bilstm(x)
        upd_velocity = torch.sigmoid(self.velo_fc(x))
        return {'velocity_output': upd_velocity}


################ ONF Session ################# Block and Module ###############
###############################################################################

class ONFConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum)
        self.init_weight()
    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
    def forward(self, input, pool_type=None):
        """Args: input: (batch_size, in_channels, time_steps, freq_bins)
        Outputs: output: (batch_size, out_channels, classes_num)"""
        x = F.relu_(self.bn1(self.conv1(input)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=(1, 2)) 
        return x


class OriginalModelONF2018(nn.Module):
    def __init__(self, classes_num, input_shape, momentum):
        super().__init__()
        self.conv1 = ONFConvBlock(1, 48, momentum)
        self.conv2 = ONFConvBlock(48, 48, momentum)
        self.conv3 = ONFConvBlock(48, 96, momentum)
        with torch.no_grad():
            dummy = torch.zeros((1, 1, 1000, input_shape))
            x = self.conv3(self.conv2(self.conv1(dummy)), pool_type='max')
            midfeat = x.shape[2] * x.shape[3] * x.shape[1]
        self.fc4 = nn.Linear(midfeat, 768, bias=False)
        self.bn4 = nn.BatchNorm1d(768, momentum=momentum)
        self.fc = nn.Linear(768, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_layer(self.fc4)
        init_bn(self.bn4)
        init_layer(self.fc)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x, 'max')
        x = self.conv3(x, 'max')
        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn4(self.fc4(x).transpose(1, 2)).transpose(1, 2))
        return torch.sigmoid(self.fc(x))


class Single_Velocity_ONF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        sr, fft, fps, feat, cls = (
            cfg.feature.sample_rate,
            cfg.feature.fft_size,
            cfg.feature.frames_per_second,
            cfg.feature.audio_feature,
            cfg.feature.classes_num)
        self.feature_extractor, self.FRE = get_feature_extractor_and_bins(feat, sr, fft, fps)
        self.bn0 = nn.BatchNorm2d(self.FRE, 0.01)
        self.velocity_model = OriginalModelONF2018(cls, self.FRE, 0.01)
        self.init_weight()
    def init_weight(self): init_bn(self.bn0)
    def forward(self, input):
        x = self.bn0(self.feature_extractor(input).unsqueeze(3)).transpose(1, 3)
        return {'velocity_output': self.velocity_model(x)}


class Dual_Velocity_ONF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        sr, fft, fps, feat, cls = (
            cfg.feature.sample_rate,
            cfg.feature.fft_size,
            cfg.feature.frames_per_second,
            cfg.feature.audio_feature,
            cfg.feature.classes_num)
        self.feature_extractor, self.FRE = get_feature_extractor_and_bins(feat, sr, fft, fps)
        self.bn0 = nn.BatchNorm2d(self.FRE, 0.01)
        self.velocity_model = OriginalModelONF2018(cls, self.FRE, 0.01)
        self.bilstm = nn.LSTM(88 * 2, 256, 1, batch_first=True, bidirectional=True)
        self.velo_fc = nn.Linear(512, cls, bias=True)
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
        init_bilstm(self.bilstm)
        init_layer(self.velo_fc)
    def forward(self, input1, input2):
        """TypeA/B/C 融合方式与 HPT 其他模型一致。"""
        x_feat = self.bn0(self.feature_extractor(input1).unsqueeze(3)).transpose(1, 3)
        pre_velocity = self.velocity_model(x_feat)
        x = torch.cat((pre_velocity, input2), dim=2)  # ← TypeA
        (x, _) = self.bilstm(x)
        return {'velocity_output': torch.sigmoid(self.velo_fc(x))}


class Triple_Velocity_ONF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        sr, fft, fps, feat, cls = (
            cfg.feature.sample_rate,
            cfg.feature.fft_size,
            cfg.feature.frames_per_second,
            cfg.feature.audio_feature,
            cfg.feature.classes_num)
        self.feature_extractor, self.FRE = get_feature_extractor_and_bins(feat, sr, fft, fps)
        self.bn0 = nn.BatchNorm2d(self.FRE, 0.01)
        self.velocity_model = OriginalModelONF2018(cls, self.FRE, 0.01)
        self.bilstm = nn.LSTM(88 * 3, 256, 1, batch_first=True, bidirectional=True)
        self.velo_fc = nn.Linear(512, cls, bias=True)
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
        init_bilstm(self.bilstm)
        init_layer(self.velo_fc)
    def forward(self, input1, input2, input3):
        """
        TypeA/B/C 同 HPT 三输入版本。
        """
        x_feat = self.bn0(self.feature_extractor(input1).unsqueeze(3)).transpose(1, 3)
        pre_velocity = self.velocity_model(x_feat)
        x = torch.cat((pre_velocity, input2, input3), dim=2)  # ← TypeA
        (x, _) = self.bilstm(x)
        return {'velocity_output': torch.sigmoid(self.velo_fc(x))}
