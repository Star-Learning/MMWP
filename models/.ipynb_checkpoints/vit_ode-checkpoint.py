import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
from einops import rearrange

from models.temporal_ode import ODEModel, CNNDynamics

class VisionTransformer(nn.Module):
    def __init__(self,
                 in_channels=10,  # 假设有10个输入变量
                 out_channels=10,  # 假设有10个输出变量
                 out_shape=(32, 64),
                 pretrained_model_path='/home/jovyan/work/downloads/vit/vit-base-patch16-224-in21k',
                 time_window=6,
                 pred_window=6,
                use_ode = True):
        super(VisionTransformer, self).__init__()
        
        self.in_shape = (224, 224)  # ViT 预训练模型的输入大小
        self.out_shape = out_shape
        self.time_window = time_window
        self.pred_window = pred_window
        self.out_channels = out_channels
        
        self.use_ode = use_ode
        self.cnn_dynamics = CNNDynamics(c=in_channels)
        self.ode_model = ODEModel(time_steps=time_window, pde=self.cnn_dynamics, time_interval=1.0)

        self.conv = nn.Conv2d(in_channels * time_window, 3, kernel_size=1)
        
        self.vit_model = ViTModel.from_pretrained(pretrained_model_path)
        self.patch_size = self.vit_model.config.patch_size
        self.hidden_size = self.vit_model.config.hidden_size
        
        self.regression_head = nn.Conv2d(self.hidden_size, out_channels, kernel_size=1)

    def forward(self, x):
        batch_size, T, C, H, W = x.size()
        
    
        if self.use_ode:
            x = self.ode_model(x) # batch_size, T, C, H, W
        
        x = rearrange(x, 'b t c h w -> b (t c) h w')  #  (batch_size, T*C, h, w)
        x = F.interpolate(x, size=self.in_shape, mode='bilinear', align_corners=False)  # (batch_size, T*C, 224, 224)
        x = self.conv(x)  # (batch_size, 3, 224, 224)
        
        # Step 2: 提取 ViT 输出 (batch_size, seq_len, hidden_size)
        vit_output = self.vit_model(x).last_hidden_state  # ViT 的输出
        
        # Step 3: 去除 [CLS] 标记，仅保留 patch embeddings
        patch_embeddings = vit_output[:, 1:, :]  # (batch_size, seq_len-1, hidden_size)
        
        # Step 4: 恢复 2D 布局
        h_patches = self.in_shape[0] // self.patch_size  # patch 数量（高度）
        w_patches = self.in_shape[1] // self.patch_size  # patch 数量（宽度）
        patch_embeddings = patch_embeddings.view(batch_size, h_patches, w_patches, self.hidden_size)  # (batch_size, h', w', hidden_size)
        patch_embeddings = patch_embeddings.permute(0, 3, 1, 2)  # 调整为卷积输入格式 (batch_size, hidden_size, h', w')
        
        # Step 5: 使用回归头生成多通道输出
        x = self.regression_head(patch_embeddings)  # (batch_size, out_channels, h', w')
        
        # Step 6: 恢复到目标大小
        x = F.interpolate(x, size=self.out_shape, mode='bilinear', align_corners=False)  # (batch_size, out_channels, h, w)
        x = x.unsqueeze(1).repeat(1, self.pred_window, 1, 1, 1)  # (batch_size, T', out_channels, h, w)
        
        return x  # 输出 (batch_size, T', out_channels, h, w)

# 示例使用
if __name__ == "__main__":
    # 假设我们有4个输入通道（2个变量各2通道，1个变量1通道），4个输出通道
    in_channels = 42
    out_channels = 1
    time_window = 4
    pred_window = 4
    out_shape = (32, 64)
    
    # 创建一个简单的输入张量 (batch_size=2, T=6, n_vars=4, H=32, W=64)
    input_tensor = torch.randn(2, time_window, in_channels, *out_shape)
    
    # 模型实例化
    model = VisionTransformer(in_channels=in_channels, out_channels=out_channels, out_shape=out_shape, time_window=time_window, pred_window=pred_window)
    
    # 前向传播
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # 应该是 (batch_size, T', out_channels, H, W) = (2, 6, 4, 32, 64)