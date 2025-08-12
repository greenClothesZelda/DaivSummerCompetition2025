import torch
import torch.nn as nn
from .modules.vit import ViT
from .modules.decoder import Decoder

class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder (MAE) model.
    This model consists of a Vision Transformer (ViT) encoder and a decoder.
    The encoder processes a subset of image patches, and the decoder reconstructs the
    full image from the encoded representation and mask tokens.
    """
    def __init__(
        self,
        encoder,
        decoder,
        masking_ratio=0.75,
    ):
        """
        Initializes the MaskedAutoencoder model.

        Args:
            encoder (nn.Module): The encoder model (typically a ViT).
            decoder (nn.Module): The decoder model.
            masking_ratio (float, optional): The ratio of patches to mask. Defaults to 0.75.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.masking_ratio = masking_ratio
        self.patch_size = encoder.patch_size
        self.embed_size = encoder.embed_size

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.embed_size))

    def forward(self, img):
        """
        Forward pass for the MaskedAutoencoder.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The reconstructed image patches.
                - The original image patches.
                - The mask indicating which patches were masked.
        """
        # Patchify the image
        patches = img.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(img.size(0), -1, 3, self.patch_size, self.patch_size)
        patches = patches.flatten(2)
        
        batch_size, num_patches, _ = patches.shape

        # Create mask
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch_size, num_patches).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # Get patches for the encoder
        batch_range = torch.arange(batch_size)[:, None]
        unmasked_patches = patches[batch_range, unmasked_indices]

        # Encode the unmasked patches
        encoded_patches = self.encoder.patch_embedding(unmasked_patches.reshape(-1, 3, self.patch_size, self.patch_size))
        encoded_patches = encoded_patches.flatten(2).transpose(1, 2)
        encoded_patches = encoded_patches.reshape(batch_size, -1, self.embed_size)
        
        # Add positional embeddings
        pos_embed = self.encoder.pos_embedding.repeat(batch_size, 1, 1)
        encoded_patches += pos_embed[batch_range, unmasked_indices + 1]

        # Create input for the decoder
        decoder_input = torch.cat([encoded_patches, self.mask_token.repeat(batch_size, num_masked, 1)], dim=1)
        
        # Add positional embeddings to all tokens for the decoder
        decoder_pos_embed = torch.cat([pos_embed[batch_range, unmasked_indices + 1], pos_embed[batch_range, masked_indices + 1]], dim=1)
        decoder_input += decoder_pos_embed

        # Decode
        decoded_patches = self.decoder(decoder_input)

        # Get the original patches for the masked areas for loss calculation
        masked_patches = patches[batch_range, masked_indices]

        return decoded_patches[:, -num_masked:], masked_patches, masked_indices

def get_mae_model(
    image_size=64,
    patch_size=8,
    encoder_layers=12,
    encoder_embed_size=768,
    encoder_heads=12,
    decoder_layers=8,
    decoder_heads=16,
    masking_ratio=0.75,
):
    """
    A helper function to create a Masked Autoencoder model with a ViT encoder.

    Args:
        image_size (int, optional): The size of the input image. Defaults to 64.
        patch_size (int, optional): The size of the image patches. Defaults to 8.
        encoder_layers (int, optional): The number of layers in the encoder. Defaults to 12.
        encoder_embed_size (int, optional): The embedding size for the encoder. Defaults to 768.
        encoder_heads (int, optional): The number of attention heads in the encoder. Defaults to 12.
        decoder_layers (int, optional): The number of layers in the decoder. Defaults to 8.
        decoder_heads (int, optional): The number of attention heads in the decoder. Defaults to 16.
        masking_ratio (float, optional): The ratio of patches to mask. Defaults to 0.75.

    Returns:
        MaskedAutoencoder: The initialized Masked Autoencoder model.
    """
    encoder = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=1000,  # Not used in MAE pre-training, but required by ViT
        num_layers=encoder_layers,
        embed_size=encoder_embed_size,
        heads=encoder_heads,
        dropout=0.1,
        forward_expansion=4,
    )

    decoder = Decoder(
        num_layers=decoder_layers,
        embed_size=encoder_embed_size,
        heads=decoder_heads,
        forward_expansion=4,
        dropout=0.1,
        patch_size=patch_size,
    )

    mae = MaskedAutoencoder(
        encoder=encoder,
        decoder=decoder,
        masking_ratio=masking_ratio,
    )

    return mae
