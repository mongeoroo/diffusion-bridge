import torch.nn as nn
import functools
import torch
import numpy as np
from .. import utils, layers, normalization


class ResnetBlock1D(nn.Module):
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
                 dropout=0.1, init_scale=1.):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.act = act
        self.dense = nn.Linear(temb_dim, out_ch) if temb_dim else None
        
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(min(in_ch // 4, 32), in_ch)
        self.norm2 = nn.GroupNorm(min(out_ch // 4, 32), out_ch)
        
        if up:
            self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            self.upsample = nn.Upsample(scale_factor=2, mode='linear')
        elif down:
            self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            self.downsample = nn.Conv1d(in_ch, out_ch, 3, stride=2, padding=1)
        else:
            self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            self.up = self.down = False

        self.dropout = nn.Dropout(dropout)
        self.init_scale = init_scale
        self.up = up
        self.down = down

    def forward(self, x, temb=None):
        h = self.act(self.norm1(x))
        if self.up:
            h = self.upsample(h)
            x = self.upsample(x)
        
        h = self.conv1(h)
        
        if temb is not None:
            h += self.dense(self.act(temb))[:, :, None]
            
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        if self.down:
            x = self.downsample(x)
            h = self.downsample(h)

        return h + self.shortcut(x)

def ncsnpp_1d(dim, init_dim, dim_mults, channels):
    """간단한 설정으로 NCSNpp 모델을 생성하는 함수"""
    class Config:
        def __init__(self):
            self.model = type('ModelConfig', (), {
                'nf': dim,
                'ch_mult': dim_mults,
                'num_res_blocks': 2,
                'embedding_type': 'positional',
                'scale_by_sigma': True
            })
            self.data = type('DataConfig', (), {
                'channels': channels
            })

    config = Config()
    return NCSNpp1D(config)

@utils.register_model(name='ncsnpp_1d')
class NCSNpp1D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = act = nn.SiLU()
        self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))
        
        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult  # 예: (1, 2, 4, 8)
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.embedding_type = config.model.embedding_type.lower()
        
        embed_dim = nf * 4
        self.embed = nn.Sequential(
            nn.Linear(nf, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.num_resolutions = num_resolutions = len(ch_mult)
        
        # Downsampling blocks
        modules = [nn.Conv1d(config.data.channels, nf, 3, padding=1)]
        hs_c = [nf]
        in_ch = nf
        
        # Down blocks
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock1D(act, in_ch, out_ch, embed_dim))
                in_ch = out_ch
                hs_c.append(in_ch)
            
            if i_level != num_resolutions - 1:
                modules.append(ResnetBlock1D(act, in_ch, in_ch, embed_dim, down=True))
                hs_c.append(in_ch)

        # Middle blocks
        modules.append(ResnetBlock1D(act, in_ch, in_ch, embed_dim))
        modules.append(ResnetBlock1D(act, in_ch, in_ch, embed_dim))

        # Up blocks
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock1D(act, in_ch + hs_c.pop(), out_ch, embed_dim))
                in_ch = out_ch

            if i_level != 0:
                modules.append(ResnetBlock1D(act, in_ch, in_ch, embed_dim, up=True))

        # End
        modules.append(nn.GroupNorm(min(in_ch // 4, 32), in_ch))
        modules.append(nn.SiLU())
        modules.append(nn.Conv1d(in_ch, config.data.channels, 3, padding=1))
        
        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond):
        modules = self.all_modules
        m_idx = 0
        
        if self.embedding_type == 'fourier':
            temb = self.embed(time_cond)
        else:
            temb = self.embed(time_cond)

        # Downsampling
        hs = [modules[m_idx](x)]
        m_idx += 1

        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                hs.append(h)

        # Middle
        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if i_level != 0:
                h = modules[m_idx](h, temb)
                m_idx += 1

        # End
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1

        assert m_idx == len(modules)
        
        if self.config.model.scale_by_sigma:
            used_sigmas = time_cond.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        return h