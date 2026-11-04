"""CLIP symmetric contrastive loss."""
import torch
import torch.nn.functional as F


def clip_loss(image_features, text_features, temperature=0.07):
    """CLIP：每个图片与对应文本为正例，对角线标签为 0,1,...,B-1。"""
    if temperature <= 0:
        raise ValueError('temperature 必须大于 0')
    if image_features.shape != text_features.shape or image_features.ndim != 2:
        raise ValueError('要求图文特征形状相同，均为 (B, D)')
    images = F.normalize(image_features, dim=-1)
    texts = F.normalize(text_features, dim=-1)
    logits = images @ texts.T / temperature  # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2


if __name__ == '__main__':
    torch.manual_seed(7)
    image_embeddings = torch.randn(4, 32, requires_grad=True)
    text_embeddings = torch.randn(4, 32, requires_grad=True)
    loss = clip_loss(image_embeddings, text_embeddings)
    print('CLIP contrastive loss:', loss.item())
    loss.backward()
    print('图像特征梯度形状:', image_embeddings.grad.shape)
