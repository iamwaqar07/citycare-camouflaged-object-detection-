
import gradio as gr
import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
from torchvision import transforms


device = torch.device("cpu")


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(8, channels // 16)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        return x * torch.sigmoid(self.mlp(avg) + self.mlp(mx))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.max(x, dim=1, keepdim=True)[0]
        attention = torch.sigmoid(
            self.conv(torch.cat([avg, mx], dim=1))
        )
        return x * attention


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True, dropout=0):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.attention = CBAM(out_channels) if use_attention else nn.Identity()

    def forward(self, x, skip):
        x = F.interpolate(
            x, size=skip.shape[2:],
            mode="bilinear", align_corners=False
        )
        x = torch.cat([x, skip], dim=1)
        return self.attention(self.conv(x))


class Res2NetCOD(nn.Module):
    def __init__(self, use_attention=True):
        super().__init__()

        self.encoder = timm.create_model(
            "res2net50_26w_4s",
            pretrained=False,
            features_only=True
        )

        self.decoder1 = DecoderBlock(3072, 512, use_attention, 16)
        self.decoder2 = DecoderBlock(1024, 256, use_attention, 16)
        self.decoder3 = DecoderBlock(512, 128, use_attention, 16)
        self.decoder4 = DecoderBlock(192, 64, use_attention, 8)

        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        input_size = x.shape[2:]
        features = self.encoder(x)

        x = self.decoder1(features[4], features[3])
        x = self.decoder2(x, features[2])
        x = self.decoder3(x, features[1])
        x = self.decoder4(x, features[0])

        x = self.final(x)

        return F.interpolate(
            x, size=input_size,
            mode="bilinear", align_corners=False
        )


# Load trained checkpoint
model = Res2NetCOD(use_attention=True)

checkpoint = torch.load(
    "best_model.pth",
    map_location="cpu",
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.to('cpu')
model.eval()


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@spaces.GPU
def predict_image(image):
    if image is None:
        return None, None

    original = image.convert("RGB")
    model.to('cuda')
    tensor = transform(original).unsqueeze(0).to('cuda')

    with torch.no_grad():
        output = torch.sigmoid(model(tensor))

    mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    mask_image = Image.fromarray(mask).resize(
        original.size, Image.Resampling.NEAREST
    )

    original_array = np.array(original)
    mask_array = np.array(mask_image) > 0

    overlay = original_array.copy()
    overlay[mask_array] = (
        0.5 * original_array[mask_array] +
        0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    model.to('cpu')
    torch.cuda.empty_cache()
    return mask_image, Image.fromarray(overlay)


demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[
        gr.Image(type="pil", label="Segmentation Mask"),
        gr.Image(type="pil", label="Detected Object Overlay")
    ],
    title="CityCare Hospital - Camouflaged Object Detection",
    description="Res2Net-50 with CBAM attention for camouflaged object detection."
)


if __name__ == "__main__":
    demo.launch()
