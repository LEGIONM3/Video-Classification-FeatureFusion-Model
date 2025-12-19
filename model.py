import torch
import torch.nn as nn
import torchvision.models.video as models

class TimeSformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_frames):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn_time = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.attn_space = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.num_frames = num_frames
        
    def forward(self, x):
        B, TP, D = x.shape
        T = self.num_frames
        P = TP // T
        
        # Temporal Attention
        xt = x.view(B, T, P, D).permute(0, 2, 1, 3).reshape(B * P, T, D)
        xt_res = xt
        xt = self.norm1(xt)
        xt, _ = self.attn_time(xt, xt, xt)
        xt = xt + xt_res
        x = xt.view(B, P, T, D).permute(0, 2, 1, 3).reshape(B, TP, D)
        
        # Spatial Attention
        xs = x.view(B, T, P, D).reshape(B * T, P, D)
        xs_res = xs
        xs = self.norm2(xs)
        xs, _ = self.attn_space(xs, xs, xs)
        xs = xs + xs_res
        x = xs.view(B, T, P, D).reshape(B, TP, D)
        
        x = x + self.mlp(self.norm3(x))
        return x

class FeatureFusionNetwork(nn.Module):
    def __init__(self):
        super(FeatureFusionNetwork, self).__init__()
        
        # Branch 1: Backbone CNN (ResNet3D)
        self.cnn = models.r3d_18(weights=None)
        self.cnn.fc = nn.Identity() # Output 512
        
        # Branch 2: TimeSformer Backbone
        self.patch_size = 16
        self.embed_dim = 256
        self.img_size = 112
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.num_frames = 16 # Default SEQ_LEN
        
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_frames * self.num_patches + 1, self.embed_dim))
        
        self.transformer_layer = TimeSformerBlock(self.embed_dim, num_heads=4, num_frames=self.num_frames)
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(512 + self.embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        
    def forward(self, x):
        # CNN Pathway
        cnn_feat = self.cnn(x) # (B, 512)
        
        # Transformer Pathway
        b, c, t, h, w = x.shape
        x_uv = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        patches = self.patch_embed(x_uv).flatten(2).transpose(1, 2)
        patches = patches.reshape(b, t * self.num_patches, self.embed_dim)
        
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x_trans = torch.cat((cls_tokens, patches), dim=1)
        x_trans = x_trans + self.pos_embed[:, :x_trans.size(1), :]
        
        patch_tokens = x_trans[:, 1:, :]
        out_patches = self.transformer_layer(patch_tokens)
        trans_feat = out_patches.mean(dim=1) # (B, D)
        
        combined = torch.cat((cnn_feat, trans_feat), dim=1)
        out = self.fusion_fc(combined)
        return out
