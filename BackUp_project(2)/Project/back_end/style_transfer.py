"""
Neural Style Transfer using VGG19 (Gatys et al. approach)
Works on CPU — no GPU/CUDA required.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy
import time
from typing import Optional, Callable


# ──────────────────────────────────────────────
#  DEVICE CONFIGURATION (CPU only)
# ──────────────────────────────────────────────
DEVICE = torch.device("cpu")


# ──────────────────────────────────────────────
#  IMAGE LOADING / SAVING UTILITIES
# ──────────────────────────────────────────────
def load_image(image_path: str, max_size: int = 512) -> torch.Tensor:
    """Load an image, resize it, and convert to a normalized tensor."""
    image = Image.open(image_path).convert("RGB")

    # Resize keeping aspect ratio, longest side = max_size
    ratio = max_size / max(image.size)
    new_size = tuple([int(round(d * ratio)) for d in image.size])
    image = image.resize(new_size, Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # VGG19 normalization
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Add batch dimension: (1, C, H, W)
    tensor = transform(image).unsqueeze(0)
    return tensor.to(DEVICE)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized tensor back to a PIL Image."""
    image = tensor.clone().detach().squeeze(0)  # remove batch dim

    # Undo VGG normalization
    denormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    image = denormalize(image)
    image = image.clamp(0, 1)

    # Convert to PIL
    to_pil = transforms.ToPILImage()
    return to_pil(image)


# ──────────────────────────────────────────────
#  LOSS MODULES
# ──────────────────────────────────────────────
class ContentLoss(nn.Module):
    """Computes MSE loss between current features and target content features."""

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target.detach()
        self.loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.loss = nn.functional.mse_loss(x, self.target)
        return x  # pass through


class GramMatrix(nn.Module):
    """Compute the Gram matrix for style representation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.size()
        features = x.view(batch * channels, height * width)
        gram = torch.mm(features, features.t())
        # Normalize
        return gram / (batch * channels * height * width)


class StyleLoss(nn.Module):
    """Computes MSE loss between Gram matrices of current and target style features."""

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.gram = GramMatrix()
        self.target = self.gram(target).detach()
        self.loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gram_x = self.gram(x)
        self.loss = nn.functional.mse_loss(gram_x, self.target)
        return x  # pass through


# ──────────────────────────────────────────────
#  BUILD THE STYLE TRANSFER MODEL
# ──────────────────────────────────────────────

# Layers where we extract content and style features
CONTENT_LAYERS = ["conv_4"]
STYLE_LAYERS = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]


def build_model(
    content_img: torch.Tensor,
    style_img: torch.Tensor,
):
    """
    Build a modified VGG19 that has ContentLoss and StyleLoss
    modules inserted at the right places.
    """
    # Load pretrained VGG19 features and freeze them
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(DEVICE).eval()
    for param in vgg.parameters():
        param.requires_grad_(False)

    model = nn.Sequential()
    content_losses = []
    style_losses = []

    # We need a normalization layer is already handled in load_image
    conv_count = 0

    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            conv_count += 1
            name = f"conv_{conv_count}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{conv_count}"
            layer = nn.ReLU(inplace=False)  # out-of-place for correctness
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{conv_count}"
            # Average pooling often gives smoother results
            layer = nn.AvgPool2d(kernel_size=2, stride=2)
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{conv_count}"
        else:
            name = f"other_{conv_count}"

        model.add_module(name, layer)

        # Insert content loss
        if name in CONTENT_LAYERS:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{conv_count}", content_loss)
            content_losses.append(content_loss)

        # Insert style loss
        if name in STYLE_LAYERS:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module(f"style_loss_{conv_count}", style_loss)
            style_losses.append(style_loss)

    # Trim layers after the last loss module (no need to compute further)
    last_loss_idx = 0
    for i, (name, _) in enumerate(model.named_children()):
        if "content_loss" in name or "style_loss" in name:
            last_loss_idx = i

    # Rebuild trimmed model
    trimmed = nn.Sequential()
    for i, (name, layer) in enumerate(model.named_children()):
        trimmed.add_module(name, layer)
        if i == last_loss_idx:
            break

    return trimmed, content_losses, style_losses


# ──────────────────────────────────────────────
#  RUN STYLE TRANSFER
# ──────────────────────────────────────────────
def run_style_transfer(
    content_path: str,
    style_path: str,
    output_path: str,
    image_size: int = 384,
    num_steps: int = 200,
    style_weight: float = 1_000_000,
    content_weight: float = 1,
    progress_callback: Optional[Callable] = None,
) -> str:
    """
    Perform neural style transfer.

    Args:
        content_path:  Path to the content image.
        style_path:    Path to the style image.
        output_path:   Where to save the result.
        image_size:    Max dimension of the images (smaller = faster on CPU).
        num_steps:     Number of optimization iterations.
        style_weight:  How much to weigh style (higher = more stylized).
        content_weight: How much to weigh content preservation.
        progress_callback: Optional fn(step, total, loss) for progress updates.

    Returns:
        Path to the saved output image.
    """
    print(f"[StyleTransfer] Loading images (max_size={image_size})...")
    content_img = load_image(content_path, max_size=image_size)
    style_img = load_image(style_path, max_size=image_size)

    # Style image must match content image dimensions for the model
    style_img = nn.functional.interpolate(
        style_img,
        size=content_img.shape[2:],
        mode="bilinear",
        align_corners=False,
    )

    # Initialize the generated image as a copy of the content image
    generated = content_img.clone().requires_grad_(True)

    print("[StyleTransfer] Building model...")
    model, content_losses, style_losses = build_model(content_img, style_img)

    # Use L-BFGS optimizer (works best for style transfer)
    optimizer = optim.LBFGS([generated], max_iter=1, lr=1.0)

    print(f"[StyleTransfer] Running {num_steps} optimization steps on CPU...")
    start_time = time.time()

    step_count = [0]  # mutable container for closure

    while step_count[0] < num_steps:
        def closure():
            # Clamp pixel values
            with torch.no_grad():
                generated.clamp_(-3, 3)  # roughly valid range after normalization

            optimizer.zero_grad()
            model(generated)

            # Compute losses
            c_loss = sum(cl.loss for cl in content_losses)
            s_loss = sum(sl.loss for sl in style_losses)

            total_loss = content_weight * c_loss + style_weight * s_loss
            total_loss.backward()

            step_count[0] += 1

            if step_count[0] % 25 == 0 or step_count[0] == 1:
                elapsed = time.time() - start_time
                print(
                    f"  Step {step_count[0]:>4}/{num_steps} | "
                    f"Content: {c_loss.item():>10.2f} | "
                    f"Style: {s_loss.item():>10.6f} | "
                    f"Total: {total_loss.item():>12.2f} | "
                    f"Time: {elapsed:.1f}s"
                )

            if progress_callback:
                progress_callback(step_count[0], num_steps, total_loss.item())

            return total_loss

        optimizer.step(closure)

    # Final clamp and save
    with torch.no_grad():
        generated.clamp_(-3, 3)

    elapsed = time.time() - start_time
    print(f"[StyleTransfer] Done in {elapsed:.1f} seconds")

    output_image = tensor_to_image(generated)
    output_image.save(output_path, quality=95)
    print(f"[StyleTransfer] Saved to {output_path}")

    return output_path


# ──────────────────────────────────────────────
#  QUICK TEST
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python style_transfer.py <content.jpg> <style.jpg> <output.jpg>")
        sys.exit(1)

    run_style_transfer(
        content_path=sys.argv[1],
        style_path=sys.argv[2],
        output_path=sys.argv[3],
        image_size=256,
        num_steps=150,
    )