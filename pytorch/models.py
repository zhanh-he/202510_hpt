import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from pytorch_utils import move_data_to_device
from mamba_ssm import Mamba
from einops import rearrange

class Rearrange(nn.Module):
    def __init__(self, pattern):
        super(Rearrange, self).__init__()
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
    """Initialize a GRU layer. """
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in: (i + 1) * fan_in, :])
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)

################ Mamba Session ################# Block and Module ##############
################################################################################

class Mamba1(nn.Module):
    def __init__(self, classes_num, midfeat, momentum):
        super(Mamba1, self).__init__()
        self.conv_block1 = HPTConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = HPTConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = HPTConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = HPTConvBlock(in_channels=96, out_channels=128, momentum=momentum)
        self.fc5 = nn.Linear(in_features=midfeat, out_features=768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)
        self.mamba = Mamba( # Replace GRU with Mamba
            d_model=768,    # Model dimension
            d_state=16,     # SSM state expansion factor
            d_conv=4,       # Local convolution width
            expand=2)       # Block expansion factor
        self.layer_norm = nn.LayerNorm(768)  # Normalize before Mamba
        self.proj = nn.Linear(in_features=768, out_features=classes_num, bias=True)
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_layer(self.proj)

    def forward(self, input):
        x = self.conv_block1(input)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.transpose(1, 2).flatten(2)  # Transform for linear layer
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training)

        # Use Mamba directly
        x = self.layer_norm(x)
        x = self.mamba(x)
        x = F.dropout(x, p=0.5, training=self.training)

        output = torch.sigmoid(self.proj(x))
        return output


class Mamba2(nn.Module):
    def __init__(self, classes_num, midfeat, momentum):
        super(Mamba2, self).__init__()
        self.conv_block1 = HPTConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = HPTConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = HPTConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = HPTConvBlock(in_channels=96, out_channels=128, momentum=momentum)
        self.fc5 = nn.Linear(in_features=midfeat, out_features=768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)
        
        mamba_insize=768
        self.layer_norm = nn.LayerNorm(mamba_insize)  # Normalize before Mamba
        self.mamba = Mamba( # Replace GRU with Mamba
            d_model=mamba_insize,    # Model dimension
            d_state=16,     # SSM state expansion factor
            d_conv=4,       # Local convolution width
            expand=2)       # Block expansion factor
        self.mamba_layer = nn.Sequential(
            nn.LayerNorm(768), self.mamba) # mamba input_size
        self.conv_layer = nn.Sequential(nn.LayerNorm(mamba_insize),
                                Rearrange('b n c -> b c n'),
                                nn.Conv1d(mamba_insize, mamba_insize, kernel_size=31,groups=mamba_insize,padding='same'),
                                Rearrange('b c n -> b n c')
                                )
        self.proj = nn.Linear(in_features=mamba_insize, out_features=classes_num, bias=True)
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_layer(self.proj)

    def forward(self, input):
        x = self.conv_block1(input)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.transpose(1, 2).flatten(2)  # Transform for linear layer
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training)

        # Use Mamba - FasNet Method
        mamba_input = self.layer_norm(x)
        mamba_output = self.mamba(mamba_input)
        mamba_output = mamba_input + self.mamba_layer(mamba_input)
        output = self.conv_layer(mamba_output)
        mamba_output = self.proj(mamba_output.contiguous().view(-1, mamba_output.shape[2])).view(mamba_input.shape)
        output = output + mamba_output
        return output

class Single_Velo_Mamba1(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Single_Velo_Mamba1, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate//2, sample_rate//frames_per_second # 16000Hz/100fps = 160steps/s
        midfeat, momentum = 1792, 0.01
        # Log Mel Spectrogram extractor
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size, win_length=window_size,
                                                 center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = Mamba1(classes_num, midfeat, momentum)  # OriginalModelHPT2020
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
    def forward(self, input):
        """Args: input: (batch_size, data_length)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}
        """        # batch=12, ch=1, timsteps=10sx100steps/s=1001steps, melbins=229 (old version, torchlibrosa)
        x = self.logmel_extractor(input)  	# batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x = x.unsqueeze(3)                  # batch=12, melbins=229, timsteps=1001, ch=1
        x = self.bn0(x)					    # batch=12, melbins=229, timsteps=1001, ch=1
        x = x.transpose(1, 3)				# batch=12, ch=1, timsteps=1001, melbins=229
        est_velocity = self.velocity_model(x)  	# batch=12, timsteps=1001, classes_num=88
        output_dict = {'velocity_output': est_velocity}
        return output_dict


class Single_Velo_Mamba2(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Single_Velo_Mamba2, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate//2, sample_rate//frames_per_second # 16000Hz/100fps = 160steps/s
        midfeat, momentum = 1792, 0.01
        # Log Mel Spectrogram extractor
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size, win_length=window_size,
                                                 center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = Mamba2(classes_num, midfeat, momentum)  # OriginalModelHPT2020
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
    def forward(self, input):
        """Args: input: (batch_size, data_length)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}
        """        # batch=12, ch=1, timsteps=10sx100steps/s=1001steps, melbins=229 (old version, torchlibrosa)
        x = self.logmel_extractor(input)  	# batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x = x.unsqueeze(3)                  # batch=12, melbins=229, timsteps=1001, ch=1
        x = self.bn0(x)					    # batch=12, melbins=229, timsteps=1001, ch=1
        x = x.transpose(1, 3)				# batch=12, ch=1, timsteps=1001, melbins=229
        est_velocity = self.velocity_model(x)  	# batch=12, timsteps=1001, classes_num=88
        output_dict = {'velocity_output': est_velocity}
        return output_dict

################ HPT Session ################# Block and Module ################
################################################################################

class HPTConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        super(HPTConvBlock, self).__init__()
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

class OriginalModelHPT2020(nn.Module):
    def __init__(self, classes_num, midfeat, momentum):
        super(OriginalModelHPT2020, self).__init__()
        self.conv_block1 = HPTConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = HPTConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = HPTConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = HPTConvBlock(in_channels=96, out_channels=128, momentum=momentum)
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
        super(ModifiedModelHPT2020, self).__init__()
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

################ HPT Session ################# Single #########################
###############################################################################

class Single_Velocity_HPT(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Single_Velocity_HPT, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate//2, sample_rate//frames_per_second # 16000Hz/100fps = 160steps/s
        midfeat, momentum = 1792, 0.01
        # Log Mel Spectrogram extractor
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size, win_length=window_size,
                                                 center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = OriginalModelHPT2020(classes_num, midfeat, momentum)  # OriginalModelHPT2020
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
    def forward(self, input):
        """Args: input: (batch_size, data_length)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}
        """        # batch=12, ch=1, timsteps=10sx100steps/s=1001steps, melbins=229 (old version, torchlibrosa)
        x = self.logmel_extractor(input)  	# batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x = x.unsqueeze(3)                  # batch=12, melbins=229, timsteps=1001, ch=1
        x = self.bn0(x)					    # batch=12, melbins=229, timsteps=1001, ch=1
        x = x.transpose(1, 3)				# batch=12, ch=1, timsteps=1001, melbins=229
        est_velocity = self.velocity_model(x)  	# batch=12, timsteps=1001, classes_num=88
        output_dict = {'velocity_output': est_velocity}
        return output_dict


class Single_Velocity_HPT2(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Single_Velocity_HPT2, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate//2, sample_rate//frames_per_second # 16000Hz/100fps = 160steps/s
        midfeat, momentum = 1792, 0.01
        # Log Mel Spectrogram extractor
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size, win_length=window_size,
                                                 center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = ModifiedModelHPT2020(classes_num, midfeat, momentum)  # OriginalModelHPT2020
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
    def forward(self, input):
        """Args: input: (batch_size, data_length)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}
        """        # batch=12, ch=1, timsteps=10sx100steps/s=1001steps, melbins=229 (old version, torchlibrosa)
        x = self.logmel_extractor(input)  	# batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x = x.unsqueeze(3)                  # batch=12, melbins=229, timsteps=1001, ch=1
        x = self.bn0(x)					    # batch=12, melbins=229, timsteps=1001, ch=1
        x = x.transpose(1, 3)				# batch=12, ch=1, timsteps=1001, melbins=229
        est_velocity = self.velocity_model(x)  	# batch=12, timsteps=1001, classes_num=88
        output_dict = {'velocity_output': est_velocity}
        return output_dict

################ HPT Session ################# Dual input2 #####################
################ HPT Session ################# Dual input2 #####################

class Dual_Velocity_HPT(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Dual_Velocity_HPT, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate//2, sample_rate//frames_per_second # 16000Hz/100fps = 160steps/s
        midfeat, momentum = 1792, 0.01
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size, win_length=window_size,
                                                 center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = OriginalModelHPT2020(classes_num, midfeat, momentum)  # OriginalModelHPT2020
        self.bilstm = nn.LSTM(input_size=88 * 2, hidden_size=256, num_layers=1, bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.velo_fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
        init_bilstm(self.bilstm)
        init_layer(self.velo_fc)
    def forward(self, input1, input2):
        """Args: input1: (batch_size, data_length)
                 input2: (batch_size, time_steps, classes_num)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}
        """        # batch=12, ch=1, timsteps=10sx100steps/s=1001steps, melbins=229 (old version, torchlibrosa)
        x1 = self.logmel_extractor(input1)  		# batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x1 = x1.unsqueeze(3)			        	# batch=12, melbins=229, timsteps=1001, ch=1
        x1 = self.bn0(x1)			            	# batch=12, melbins=229, timsteps=1001, ch=1
        x1 = x1.transpose(1, 3)			    	    # batch=12, ch=1, timsteps=1001, melbins=229
        pre_velocity = self.velocity_model(x1) 	    # batch_size=12, time_steps=1001, classes_num=88
        # Use direct-in, or use onset (regression) to condition velocities
        x = torch.cat((pre_velocity, input2), dim=2)
        (x, _) = self.bilstm(x)			            
        # x = F.dropout(x, p=0.5, training=self.training) # NEW
        upd_velocity = torch.sigmoid(self.velo_fc(x))
        output_dict = {'velocity_output': upd_velocity}
        return output_dict


class Dual_Velocity_HPT_B(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Dual_Velocity_HPT_B, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate // 2, sample_rate // frames_per_second  # 16000Hz/100fps = 160steps/s
        midfeat, momentum = 1792, 0.01
        # Log Mel Spectrogram extractor
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size, win_length=window_size,
                                                 center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = OriginalModelHPT2020(classes_num, midfeat, momentum)  # OriginalModelHPT2020
        self.bilstm = nn.LSTM(input_size=88 * 2, hidden_size=256, num_layers=1, bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.velo_fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
        init_bilstm(self.bilstm)
        init_layer(self.velo_fc)
    def forward(self, input1, input2):
        """Args: input1: (batch_size, data_length)
                 input2: (batch_size, time_steps, classes_num)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}"""
        # batch=12, ch=1, timsteps=10sx100steps/s=1001steps, melbins=229 (old version, torchlibrosa)
        x1 = self.logmel_extractor(input1)      # batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x1 = x1.unsqueeze(3)                    # batch=12, melbins=229, timsteps=1001, ch=1
        x1 = self.bn0(x1)                       # batch=12, melbins=229, timsteps=1001, ch=1
        x1 = x1.transpose(1, 3)                 # batch=12, ch=1, timsteps=1001, melbins=229
        pre_velocity = self.velocity_model(x1)  # batch_size=12, time_steps=1001, classes_num=88
        x = torch.cat((pre_velocity, (pre_velocity.detach() ** 0.5) * input2), dim=2)
        (x, _) = self.bilstm(x)                 # x = F.dropout(x, p=0.5, training=self.training)
        upd_velocity = torch.sigmoid(self.velo_fc(x))
        output_dict = {'velocity_output': upd_velocity}
        return output_dict

class Dual_Velocity_HPT_C(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Dual_Velocity_HPT_C, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate // 2, sample_rate // frames_per_second  # 16000Hz/100fps = 160steps/s
        midfeat, momentum = 1792, 0.01
        # Log Mel Spectrogram extractor
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size, win_length=window_size,
                                                 center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = OriginalModelHPT2020(classes_num, midfeat, momentum)  # OriginalModelHPT2020
        self.bilstm = nn.LSTM(input_size=88 * 2, hidden_size=256, num_layers=1, bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.velo_fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
        init_bilstm(self.bilstm)
        init_layer(self.velo_fc)
    def forward(self, input1, input2):
        """Args: input1: (batch_size, data_length)
                 input2: (batch_size, time_steps, classes_num)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}
        """  # batch=12, ch=1, timsteps=10sx100steps/s=1001steps, melbins=229 (old version, torchlibrosa)
        x1 = self.logmel_extractor(input1)      # batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x1 = x1.unsqueeze(3)                    # batch=12, melbins=229, timsteps=1001, ch=1
        x1 = self.bn0(x1)                       # batch=12, melbins=229, timsteps=1001, ch=1
        x1 = x1.transpose(1, 3)                 # batch=12, ch=1, timsteps=1001, melbins=229
        pre_velocity = self.velocity_model(x1)  # batch_size=8, time_steps=1001, classes_num=88
        x = torch.cat((pre_velocity, ((pre_velocity.detach()+input2) ** 0.5) * input2), dim=2) # TypeC
        (x, _) = self.bilstm(x)                 # x = F.dropout(x, p=0.5, training=self.training)
        upd_velocity = torch.sigmoid(self.velo_fc(x))
        output_dict = {'velocity_output': upd_velocity}
        return output_dict

################ HPT Session ################# Triple input2 input3 #################
################ HPT Session ################# Triple input2 input3 #################
################ HPT Session ################# Triple input2 input3 #################

class Triple_Velocity_HPT(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Triple_Velocity_HPT, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate // 2, sample_rate // frames_per_second  # 16000Hz/100fps = 160steps/s
        midfeat_onf, momentum = 1792, 0.01
        # Log Mel Spectrogram extractor
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size, win_length=window_size,
                                                 center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = OriginalModelHPT2020(classes_num, midfeat_onf, momentum)
        self.bilstm = nn.LSTM(input_size=88 * 3, hidden_size=256, num_layers=1, bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.velo_fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
        init_bilstm(self.bilstm)
        init_layer(self.velo_fc)
    def forward(self, input1, input2, input3):
        """Args: input1: (batch_size, data_length)
                 input2:(batch_size, time_steps, classes_num)
                 input3:(batch_size, time_steps, classes_num)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}"""
        x1 = self.logmel_extractor(input1)      # batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x1 = x1.unsqueeze(3)                    # batch=12, melbins=229, timsteps=1001, ch=1
        x1 = self.bn0(x1)                       # batch=12, melbins=229, timsteps=1001, ch=1
        x1 = x1.transpose(1, 3)                 # batch=12, ch=1, timsteps=1001, melbins=229
        pre_velocity = self.velocity_model(x1)  # batch_size, time_steps, classes_num
        x = torch.cat((pre_velocity, input2, input3), dim=2)
        (x, _) = self.bilstm(x)
        # x = F.dropout(x, p=0.5, training=self.training) # NEW 
        upd_velocity = torch.sigmoid(self.velo_fc(x))
        output_dict = {'velocity_output': upd_velocity}
        return output_dict


class Triple_Velocity_HPT_B(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Triple_Velocity_HPT_B, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate // 2, sample_rate // frames_per_second  # 16000Hz/100fps = 160steps/s
        midfeat_onf, momentum = 1792, 0.01
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = OriginalModelHPT2020(classes_num, midfeat_onf, momentum)
        self.bilstm = nn.LSTM(input_size=88 * 3, hidden_size=256, num_layers=1, bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.velo_fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
        init_bilstm(self.bilstm)
        init_layer(self.velo_fc)
    def forward(self, input1, input2, input3):
        """Args: input1: (batch_size, data_length)
                 input2:(batch_size, time_steps, classes_num)
                 input3:(batch_size, time_steps, classes_num)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}"""
        x1 = self.logmel_extractor(input1)      # batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x1 = x1.unsqueeze(3)                    # batch=12, melbins=229, timsteps=1001, ch=1
        x1 = self.bn0(x1)                       # batch=12, melbins=229, timsteps=1001, ch=1
        x1 = x1.transpose(1,3)                  # batch=12, ch=1, timsteps=1001, melbins=229
        pre_velocity = self.velocity_model(x1)  # batch_size, time_steps, classes_num
        x = torch.cat((pre_velocity, (pre_velocity.detach()**0.5)*input2, input3), dim=2)
        (x, _) = self.bilstm(x)                 # x = F.dropout(x, p=0.5, training=self.training)
        upd_velocity = torch.sigmoid(self.velo_fc(x))
        output_dict = {'velocity_output': upd_velocity}
        return output_dict


class Triple_Velocity_HPT_C(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Triple_Velocity_HPT_C, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate // 2, sample_rate // frames_per_second  # 16000Hz/100fps = 160steps/s
        midfeat_onf, momentum = 1792, 0.01
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = OriginalModelHPT2020(classes_num, midfeat_onf, momentum)
        self.bilstm = nn.LSTM(input_size=88 * 3, hidden_size=256, num_layers=1, bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.velo_fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
        init_bilstm(self.bilstm)
        init_layer(self.velo_fc)
    def forward(self, input1, input2, input3):
        """Args: input1: (batch_size, data_length)
                 input2:(batch_size, time_steps, classes_num)
                 input3:(batch_size, time_steps, classes_num)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}"""
        x1 = self.logmel_extractor(input1)      # batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x1 = x1.unsqueeze(3)                    # batch=12, melbins=229, timsteps=1001, ch=1
        x1 = self.bn0(x1)                       # batch=12, melbins=229, timsteps=1001, ch=1
        x1 = x1.transpose(1,3)                  # batch=12, ch=1, timsteps=1001, melbins=229
        pre_velocity = self.velocity_model(x1)  # batch_size, time_steps, classes_num
        x = torch.cat((pre_velocity, ((pre_velocity.detach()+input2) ** 0.5) * input2, input3), dim=2)
        (x, _) = self.bilstm(x)                 # x = F.dropout(x, p=0.5, training=self.training)
        upd_velocity = torch.sigmoid(self.velo_fc(x))
        output_dict = {'velocity_output': upd_velocity}
        return output_dict


################ ONF Session ################# Block and Module ###############
###############################################################################

class ONFConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        super(ONFConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
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
    def __init__(self, classes_num, midfeat_onf, momentum):
        super(OriginalModelONF2018, self).__init__()
        self.conv_block1 = ONFConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = ONFConvBlock(in_channels=48, out_channels=48, momentum=momentum)
        self.conv_block3 = ONFConvBlock(in_channels=48, out_channels=96, momentum=momentum)
        self.fc4 = nn.Linear(midfeat_onf, 768, bias=False)
        self.bn4 = nn.BatchNorm1d(768, momentum=momentum)
        self.fc = nn.Linear(768, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_layer(self.fc4)
        init_bn(self.bn4)
        init_layer(self.fc)
    def forward(self, input):
        """Args: input: (batch_size, channels_num, time_steps, freq_bins)
        Outputs: output: (batch_size, time_steps, classes_num)"""							
        # input batch=8, chn=1,  timstep=1001, melbins=229
        x = self.conv_block1(input)		# conv1 batch=8, chn=48, timstep=1001, melbins=229
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_type='max')		# conv2 batch=8, chn=48, timstep=1001, melbins=114	
        x = self.conv_block3(x, pool_type='max')		# conv3 batch=8, chn=96, timstep=1001, melbins=57
        x = x.transpose(1, 2).flatten(2)			# 8,96,1001,57--trans--8,1001,96,57--flat--8,1001,5472
        x = F.relu(self.bn4(self.fc4(x).transpose(1, 2)).transpose(1, 2))
        # x = F.dropout(x, p=0.5, training=self.training)
        output = torch.sigmoid(self.fc(x))
        return output


class ModifiedModelONF2018(nn.Module):
    def __init__(self, classes_num, midfeat_onf, momentum):
        super(ModifiedModelONF2018, self).__init__()
        self.conv_block1 = ONFConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = ONFConvBlock(in_channels=48, out_channels=48, momentum=momentum)
        self.conv_block3 = ONFConvBlock(in_channels=48, out_channels=96, momentum=momentum)
        self.fc4 = nn.Linear(midfeat_onf, 768, bias=False)
        self.bn4 = nn.BatchNorm1d(768, momentum=momentum)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=1,  # num_layers = 2
                              bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_layer(self.fc4)
        init_bn(self.bn4)
        init_bilstm(self.bilstm)
        init_layer(self.fc)
    def forward(self, input):
        """Args: input: (batch_size, channels_num, time_steps, freq_bins)
        Outputs: output: (batch_size, time_steps, classes_num)
        Selection: x = F.dropout(x, p=0.2 or 0.5, training=self.training)"""	
        x = self.conv_block1(input)
        x = self.conv_block2(x, pool_type='max')
        x = self.conv_block3(x, pool_type='max')
        x = x.transpose(1, 2).flatten(2)
        x = F.relu(self.bn4(self.fc4(x).transpose(1, 2)).transpose(1, 2))
        (x, _) = self.bilstm(x)
        output = torch.sigmoid(self.fc(x))
        return output

################ ONF Session ################# Single ###############
#####################################################################

class Single_Velocity_ONF(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Single_Velocity_ONF, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate//2, sample_rate//frames_per_second # 16000Hz/100fps = 160steps/s
        midfeat_onf, momentum = 5472, 0.01
        # Log Mel Spectrogram extractor
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size, win_length=window_size,
                                                 center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = OriginalModelONF2018(classes_num, midfeat_onf, momentum)  # OriginalModelHPT2020
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
    def forward(self, input):
        """Args: input: (batch_size, data_length)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}"""        
        # batch=12, ch=1, timsteps=10sx100steps/s=1001steps, melbins=229 (old version, torchlibrosa)
        x = self.logmel_extractor(input)  		# batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x = x.unsqueeze(3)				# batch=12, melbins=229, timsteps=1001, ch=1
        x = self.bn0(x)					# batch=12, melbins=229, timsteps=1001, ch=1
        x = x.transpose(1, 3)				# batch=12, ch=1, timsteps=1001, melbins=229
        est_velocity = self.velocity_model(x) 	# batch_size, time_steps, classes_num
        output_dict = {'velocity_output': est_velocity}
        return output_dict


class Single_Velocity_ONF2(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Single_Velocity_ONF2, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate//2, sample_rate//frames_per_second # 16000Hz/100fps = 160steps/s
        midfeat_onf, momentum = 5472, 0.01
        # Log Mel Spectrogram extractor
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size, win_length=window_size,
                                                 center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = ModifiedModelONF2018(classes_num, midfeat_onf, momentum)  # OriginalModelHPT2020
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
    def forward(self, input):
        """Args: input: (batch_size, data_length)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}"""        
        # batch=12, ch=1, timsteps=10sx100steps/s=1001steps, melbins=229 (old version, torchlibrosa)
        x = self.logmel_extractor(input)  		# batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x = x.unsqueeze(3)				# batch=12, melbins=229, timsteps=1001, ch=1
        x = self.bn0(x)					# batch=12, melbins=229, timsteps=1001, ch=1
        x = x.transpose(1, 3)				# batch=12, ch=1, timsteps=1001, melbins=229
        est_velocity = self.velocity_model(x) 	# batch_size, time_steps, classes_num
        output_dict = {'velocity_output': est_velocity}
        return output_dict


################ ONF Session ################# Dual input2 ###############
################ ONF Session ################# Dual input2 ###############


class Dual_Velocity_ONF(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Dual_Velocity_ONF, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate//2, sample_rate//frames_per_second # 16000Hz/100fps = 160steps/s
        midfeat_onf, momentum = 5472, 0.01
        # Log Mel Spectrogram extractor
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size, win_length=window_size, center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = OriginalModelONF2018(classes_num, midfeat_onf, momentum)
        self.bilstm = nn.LSTM(input_size=88 * 2, hidden_size=256, num_layers=1, bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.velo_fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
        init_bilstm(self.bilstm)
        init_layer(self.velo_fc)
    def forward(self, input1, input2):
        """Args: input1: (batch_size, data_length)
                 input2:(batch_size, time_steps, classes_num)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}"""
        x1 = self.logmel_extractor(input1)  		# batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x1 = x1.unsqueeze(3)				# batch=12, melbins=229, timsteps=1001, ch=1
        x1 = self.bn0(x1)				# batch=12, melbins=229, timsteps=1001, ch=1
        x1 = x1.transpose(1, 3)				# batch=12, ch=1, timsteps=1001, melbins=229
        pre_velocity = self.velocity_model(x1)  	# batch_size, time_steps, classes_num
        # Use velocities to condition onset regression, maybe input2.detach() is necessary
        x = torch.cat((pre_velocity, input2), dim=2)
        (x, _) = self.bilstm(x)			# x = F.dropout(x, p=0.5, training=self.training)
        upd_velocity = torch.sigmoid(self.velo_fc(x))
        output_dict = {'velocity_output': upd_velocity}
        return output_dict


class Dual_Velocity_ONF_B(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Dual_Velocity_ONF_B, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate // 2, sample_rate // frames_per_second  # 16000Hz/100fps = 160steps/s
        midfeat_onf, momentum = 5472, 0.01
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size, win_length=window_size,
                                                 center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = OriginalModelONF2018(classes_num, midfeat_onf, momentum)
        self.bilstm = nn.LSTM(input_size=88 * 2, hidden_size=256, num_layers=1, bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.velo_fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
        init_bilstm(self.bilstm)
        init_layer(self.velo_fc)
    def forward(self, input1, input2):      # use velocities to condition onset mask
        """Args: input1: (batch_size, data_length)
                 input2:(batch_size, time_steps, classes_num)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}"""
        x1 = self.logmel_extractor(input1)  # batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x1 = x1.unsqueeze(3)                # batch=12, melbins=229, timsteps=1001, ch=1
        x1 = self.bn0(x1)                   # batch=12, melbins=229, timsteps=1001, ch=1
        x1 = x1.transpose(1, 3)             # batch=12, ch=1, timsteps=1001, melbins=229
        pre_velocity = self.velocity_model(x1)  # batch_size, time_steps, classes_num
        x = torch.cat((pre_velocity, (pre_velocity.detach() ** 0.5) * input2), dim=2)  # TypeB
        (x, _) = self.bilstm(x)             # x = F.dropout(x, p=0.5, training=self.training)
        upd_velocity = torch.sigmoid(self.velo_fc(x))
        output_dict = {'velocity_output': upd_velocity}
        return output_dict


class Dual_Velocity_ONF_C(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Dual_Velocity_ONF_C, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate // 2, sample_rate // frames_per_second  # 16000Hz/100fps = 160steps/s
        midfeat_onf, momentum = 5472, 0.01
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size, win_length=window_size,
                                                 center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = OriginalModelONF2018(classes_num, midfeat_onf, momentum)
        self.bilstm = nn.LSTM(input_size=88 * 2, hidden_size=256, num_layers=1, bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.velo_fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
        init_bilstm(self.bilstm)
        init_layer(self.velo_fc)
    def forward(self, input1, input2): # use velocities to condition onset mask
        """Args: input1: (batch_size, data_length)
                 input2:(batch_size, time_steps, classes_num)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}"""
        x1 = self.logmel_extractor(input1)   # batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x1 = x1.unsqueeze(3)                 # batch=12, melbins=229, timsteps=1001, ch=1
        x1 = self.bn0(x1)                    # batch=12, melbins=229, timsteps=1001, ch=1
        x1 = x1.transpose(1, 3)              # batch=12, ch=1, timsteps=1001, melbins=229
        pre_velocity = self.velocity_model(x1)  # batch_size, time_steps, classes_num
        (pre_velocity.detach() ** 0.5) * input2
        x = torch.cat((pre_velocity, ((pre_velocity.detach()+input2) **0.5) *input2), dim=2)  # TypeC
        (x, _) = self.bilstm(x)  # x = F.dropout(x, p=0.5, training=self.training)
        upd_velocity = torch.sigmoid(self.velo_fc(x))
        output_dict = {'velocity_output': upd_velocity}
        return output_dict


################ ONF Session ################# Triple input2 inpu3 ###############
################ ONF Session ################# Triple input2 inpu3 ###############
################ ONF Session ################# Triple input2 inpu3 ###############


class Triple_Velocity_ONF(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Triple_Velocity_ONF, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate // 2, sample_rate // frames_per_second  # 16000Hz/100fps = 160steps/s
        midfeat_onf, momentum = 5472, 0.01
        # Log Mel Spectrogram extractor
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size, win_length=window_size,
                                                 center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = OriginalModelONF2018(classes_num, midfeat_onf, momentum)
        self.bilstm = nn.LSTM(input_size=88 * 3, hidden_size=256, num_layers=1, bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.velo_fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
        init_bilstm(self.bilstm)
        init_layer(self.velo_fc)
    def forward(self, input1, input2, input3):
        """Args: input1: (batch_size, data_length)
                 input2:(batch_size, time_steps, classes_num)
                 input3:(batch_size, time_steps, classes_num)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}
        Selection: x = torch.cat((pre_velocity, (pre_velocity ** 0.5) * input2), dim=2)"""
        x1 = self.logmel_extractor(input1)  	# batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x1 = x1.unsqueeze(3)  			# batch=12, melbins=229, timsteps=1001, ch=1
        x1 = self.bn0(x1)  			# batch=12, melbins=229, timsteps=1001, ch=1
        x1 = x1.transpose(1, 3)  		# batch=12, ch=1, timsteps=1001, melbins=229
        pre_velocity = self.velocity_model(x1)  # batch_size, time_steps, classes_num
        # Use velocities to condition onset regression, maybe input2.detach() is unnecessary
        x = torch.cat((pre_velocity, input2, input3), dim=2)
        (x, _) = self.bilstm(x)  # x = F.dropout(x, p=0.5, training=self.training)
        upd_velocity = torch.sigmoid(self.velo_fc(x))
        output_dict = {'velocity_output': upd_velocity}
        return output_dict


class Triple_Velocity_ONF_B(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Triple_Velocity_ONF_B, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate // 2, sample_rate // frames_per_second  # 16000Hz/100fps = 160steps/s
        midfeat_onf, momentum = 5472, 0.01
        # Log Mel Spectrogram extractor
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size,
                                                 center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = OriginalModelONF2018(classes_num, midfeat_onf, momentum)
        self.bilstm = nn.LSTM(input_size=88 * 3, hidden_size=256, num_layers=1,  # num_layers=2
                              bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.velo_fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
        init_bilstm(self.bilstm)
        init_layer(self.velo_fc)
    def forward(self, input1, input2, input3):
        """Args: input1: (batch_size, data_length)
                 input2:(batch_size, time_steps, classes_num)
                 input3:(batch_size, time_steps, classes_num)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}"""
        x1 = self.logmel_extractor(input1)  # batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x1 = x1.unsqueeze(3)  # batch=12, melbins=229, timsteps=1001, ch=1
        x1 = self.bn0(x1)  # batch=12, melbins=229, timsteps=1001, ch=1
        x1 = x1.transpose(1, 3)  # batch=12, ch=1, timsteps=1001, melbins=229
        pre_velocity = self.velocity_model(x1)  # batch_size, time_steps, classes_num
        x = torch.cat((pre_velocity, (pre_velocity ** 0.5) * input2,  input3), dim=2)
        (x, _) = self.bilstm(x)  # x = F.dropout(x, p=0.5, training=self.training)
        upd_velocity = torch.sigmoid(self.velo_fc(x))
        output_dict = {'velocity_output': upd_velocity}
        return output_dict


# class Triple_Velocity_ONF_drop2(nn.Module):
#     def __init__(self, frames_per_second, classes_num):
#         super(Triple_Velocity_ONF_drop2, self).__init__()
#         sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
#         fmax, hop_size = sample_rate // 2, sample_rate // frames_per_second  # 16000Hz/100fps = 160steps/s
#         midfeat_onf, momentum = 5472, 0.01
#         # Log Mel Spectrogram extractor
#         self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size,
#                                                  win_length=window_size,
#                                                  center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin,
#                                                  f_max=fmax)
#         self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
#         self.velocity_model = OriginalModelONF2018(classes_num, midfeat_onf, momentum)
#         self.bilstm = nn.LSTM(input_size=88 * 3, hidden_size=256, num_layers=1,  # num_layers=2
#                               bias=True, batch_first=True, dropout=0., bidirectional=True)
#         self.velo_fc = nn.Linear(512, classes_num, bias=True)
#         self.init_weight()
#     def init_weight(self):
#         init_bn(self.bn0)
#         init_bilstm(self.bilstm)
#         init_layer(self.velo_fc)
#     def forward(self, input1, input2, input3):
#         """Args: input1: (batch_size, data_length)
#                  input2:(batch_size, time_steps, classes_num)
#                  input3:(batch_size, time_steps, classes_num)
#         Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}"""
#         x1 = self.logmel_extractor(input1)  # batch=12, melbins=229, timsteps=1001 (new,torchaudio)
#         x1 = x1.unsqueeze(3)  # batch=12, melbins=229, timsteps=1001, ch=1
#         x1 = self.bn0(x1)  # batch=12, melbins=229, timsteps=1001, ch=1
#         x1 = x1.transpose(1, 3)  # batch=12, ch=1, timsteps=1001, melbins=229
#         pre_velocity = self.velocity_model(x1)  # batch_size, time_steps, classes_num
#         # Use velocities to condition onset regression, maybe input2.detach() is unnecessary
#         x = torch.cat((pre_velocity, input2, input3), dim=2)
#         x = F.dropout(x, p=0.2, training=self.training)
#         (x, _) = self.bilstm(x)
#         x = F.dropout(x, p=0.2, training=self.training)
#         upd_velocity = torch.sigmoid(self.velo_fc(x))
#         output_dict = {'velocity_output': upd_velocity}
#         return output_dict

################ Bad Result Session ################# Bad Result Session  ###############
################ Bad Result Session ################# Bad Result Session  ###############
class Dual_Velocity_ONF_drop2(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Dual_Velocity_ONF_drop2, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate//2, sample_rate//frames_per_second # 16000Hz/100fps = 160steps/s
        midfeat_onf, momentum = 5472, 0.01
        # Log Mel Spectrogram extractor
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size, win_length=window_size, center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = OriginalModelONF2018(classes_num, midfeat_onf, momentum)
        self.bilstm = nn.LSTM(input_size=88 * 2, hidden_size=256, num_layers=1, bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.velo_fc = nn.Linear(512, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
        init_bilstm(self.bilstm)
        init_layer(self.velo_fc)
    def forward(self, input1, input2):
        x1 = self.logmel_extractor(input1)  # batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x1 = x1.unsqueeze(3)				# batch=12, melbins=229, timsteps=1001, ch=1
        x1 = self.bn0(x1)				    # batch=12, melbins=229, timsteps=1001, ch=1
        x1 = x1.transpose(1, 3)				# batch=12, ch=1, timsteps=1001, melbins=229
        pre_velocity = self.velocity_model(x1)  	# batch_size, time_steps, classes_num
        x = torch.cat((pre_velocity, input2), dim=2)
        x = F.dropout(x, p=0.2, training=self.training)
        (x, _) = self.bilstm(x)			
        x = F.dropout(x, p=0.2, training=self.training)
        upd_velocity = torch.sigmoid(self.velo_fc(x))
        output_dict = {'velocity_output': upd_velocity}
        return output_dict


class Dual_Velocity_ONF_droplstm(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super(Dual_Velocity_ONF_droplstm, self).__init__()
        sample_rate, window_size, mel_bins, fmin = 16000, 2048, 229, 30
        fmax, hop_size = sample_rate//2, sample_rate//frames_per_second # 16000Hz/100fps = 160steps/s
        midfeat_onf, momentum = 5472, 0.01
        # Log Mel Spectrogram extractor
        self.logmel_extractor = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size, win_length=window_size,
                                                 center=True, pad_mode="reflect", n_mels=mel_bins, f_min=fmin, f_max=fmax)
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum)
        self.velocity_model = OriginalModelONF2018(classes_num, midfeat_onf, momentum)
        self.velo_fc = nn.Linear(88*2, classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.velo_fc)
    def forward(self, input1, input2):
        """Args: input1: (batch_size, data_length)
                 input2:(batch_size, time_steps, classes_num)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}
        Selection: x = torch.cat((pre_velocity, (pre_velocity ** 0.5) * input2), dim=2)"""
        x1 = self.logmel_extractor(input1)  		# batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x1 = x1.unsqueeze(3)				# batch=12, melbins=229, timsteps=1001, ch=1
        x1 = self.bn0(x1)				# batch=12, melbins=229, timsteps=1001, ch=1
        x1 = x1.transpose(1, 3)				# batch=12, ch=1, timsteps=1001, melbins=229
        pre_velocity = self.velocity_model(x1)  	# batch_size, time_steps, classes_num
        # Use velocities to condition onset regression, maybe input2.detach() is necessary
        x = torch.cat((pre_velocity, input2), dim=2)
        upd_velocity = torch.sigmoid(self.velo_fc(x))
        output_dict = {'velocity_output': upd_velocity}
        return output_dict
