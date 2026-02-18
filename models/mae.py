import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
import einops
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import logging
logging.basicConfig(level=logging.DEBUG)

from .vit import Transformer

class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)

        self.enc_to_dec = nn.Linear(encoder_dim, self.decoder_dim)
        self.enc_to_dec_pos = nn.Linear(encoder_dim, self.decoder_dim)

        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img, masking_ratio=None, masking_strategy=None):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape
        num_patches = num_patches + self.encoder.num_registers

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        tokens += self.encoder.positional_ff(self.encoder.pos_embedding.to(device, dtype=tokens.dtype))

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        if masking_strategy is None:
            SELECTION = torch.rand(3,).argmax().item()
        else:
            SELECTION = masking_strategy

        HEIGHT=240
        WIDTH=440
        BigPatchIndex = torch.arange(0,1056)
        GroupBigPatchIndex = einops.rearrange(BigPatchIndex, '(h w) ->  h w ', h=HEIGHT//10, w=WIDTH//10)

        if SELECTION == 0:
            REDUCE=1
        elif SELECTION == 1:
            REDUCE=2
        else:
            REDUCE=4

        GroupBigPatchIndex = einops.rearrange(GroupBigPatchIndex, '(h p1) (w p2) -> (h w) (p2 p1)',p1=REDUCE,p2=REDUCE)
        localNumPatches = (HEIGHT // (REDUCE*10)) * (WIDTH // (REDUCE*10))
        
        rand_indices = torch.rand(batch, localNumPatches)
        rand_indices = rand_indices.argsort(dim = -1)
        rand_indices = GroupBigPatchIndex[rand_indices].reshape(batch,-1).to(device)

        if masking_ratio is None:
            num_masked = int(self.masking_ratio * num_patches)
        else:
            num_masked = int(masking_ratio * num_patches)

        logging.info(f"{num_masked=}")
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)
        unmasked_decoder_tokens = decoder_tokens +\
                self.enc_to_dec_pos(self.encoder.positional_ff(self.encoder.pos_embedding.to(device, dtype=tokens.dtype)[0,unmasked_indices]))

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.enc_to_dec_pos(self.encoder.positional_ff(self.encoder.pos_embedding.to(device, dtype=tokens.dtype)[0,masked_indices]))

        # concat the masked tokens to the decoder tokens and attend with decoder
        
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        return pred_pixel_values, masked_patches, masked_indices

    def reconstruct(self, img):

        with torch.no_grad():
            pred_pixel_values, masked_patches, masked_indices = self.forward(img)
            patches = self.to_patch(img)

        return patches, masked_indices, pred_pixel_values 



class MAE_decoder(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        encoder_masking_ratio = 0.75,
        decoder_masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert encoder_masking_ratio > 0 and encoder_masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        assert decoder_masking_ratio > 0 and decoder_masking_ratio < 1, 'masking ratio must be kept between 0 and 1'

        self.encoder_masking_ratio = encoder_masking_ratio
        self.decoder_masking_ratio = decoder_masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])
        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        #self.hrrr_decoder = hrrr_decoder
        #self.hrrr_num_patches, self.hrrr_decoder_dim = self.hrrr_decoder.pos_embedding.shape[-2:]
        self.hrrr_num_patches, self.hrrr_decoder_dim = 910, 192
        hrrr_pixel_values_per_patch = 10*10*2
        self.memory_to_dec = nn.Linear( encoder_dim, self.hrrr_decoder_dim)

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.decoder_pixel_to_emb = nn.Linear(hrrr_pixel_values_per_patch,self.hrrr_decoder_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 910, self.hrrr_decoder_dim))

        self.enc_to_dec = nn.Linear(self.hrrr_decoder_dim, decoder_dim) if self.hrrr_decoder_dim != decoder_dim else nn.Identity()
        self.enc_to_dec_pos = nn.Linear(self.hrrr_decoder_dim, decoder_dim) if self.hrrr_decoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        transformer_layer = TransformerDecoderLayer(d_model=decoder_dim, nhead=decoder_heads, dim_feedforward=decoder_dim*4, batch_first = True)
        self.decoder = TransformerDecoder(transformer_layer, num_layers=decoder_depth)
        self.enc_to_dec2 = nn.Linear(self.hrrr_decoder_dim, decoder_dim) if self.hrrr_decoder_dim != decoder_dim else nn.Identity()

        self.to_pixels = nn.Linear(decoder_dim, hrrr_pixel_values_per_patch)

        self.temporal_tokens = nn.Parameter(torch.randn(4, encoder_dim))

    def process_temporal_input(self, img, device):
        """
        Process input with multiple time steps, each having 15 channels.
        Uses a shared temporal token parameter indexed by timestep.
        
        Args:
            img: Input tensor with shape [batch, channels, height, width]
                 where channels = num_timesteps * 15
            device: Device to put tensors on
        """
        # Determine number of time steps from image channels
        batch_size, total_channels, height, width = img.shape
        n_timesteps = total_channels // 15
        
        # Initialize for collecting tokens from all time steps
        all_tokens = []
        total_patches = 0
        
        # Process each time step, from oldest to newest
        for t in range(n_timesteps):
            # Extract the 15 channels for this time step
            time_idx = n_timesteps - 1 - t  # Reverse index: newest is 0, oldest is n-1
            time_slice = img[:, t*15:(t+1)*15, ...]
            
            # Convert to patches
            patches_t = self.to_patch(time_slice)
            batch_t, num_patches_t, *_ = patches_t.shape
            
            # Add register count to patch count
            num_patches_t = num_patches_t + self.encoder.num_registers
            total_patches += num_patches_t
            
            # Convert patches to embeddings and add positional encoding
            tokens_t = self.patch_to_emb(patches_t)
            tokens_t += self.encoder.positional_ff(self.encoder.pos_embedding.to(device, dtype=tokens_t.dtype))
            
            # Add temporal token for this specific time step
            # The index accesses the appropriate row from self.temporal_tokens
            temporal_token = self.temporal_tokens[time_idx].unsqueeze(0).unsqueeze(0)
            tokens_t += temporal_token
            
            # Add to our collection, oldest first
            all_tokens.append(tokens_t)
        
        # Concatenate all tokens from oldest to newest
        tokens = torch.cat(all_tokens, dim=1)
        
        return tokens, total_patches


    def forward(self, img, img_y, two_hours=True, decoder_masking_ratio=None):

        if decoder_masking_ratio is None:
            local_decoder_masking_ratio = self.decoder_masking_ratio
        else:
            local_decoder_masking_ratio = decoder_masking_ratio

        device = img.device

        tokens, num_patches = self.process_temporal_input(img, device)
        batch = tokens.shape[0]
        batch_range = torch.arange(batch, device = device)[:, None]

        hrrr_patches = img_y

        #BEGIN EXP
        encoder_num_masked = int(self.encoder_masking_ratio * num_patches)
        encoder_rand_indices = torch.rand(batch, num_patches, device = device)

        encoder_rand_indices = encoder_rand_indices.argsort(dim = -1)

        encoder_masked_indices = encoder_rand_indices[:, :encoder_num_masked]
        encoder_unmasked_indices = encoder_rand_indices[:, encoder_num_masked:]
        #END EXP

        tokens = tokens[batch_range,encoder_unmasked_indices]

        memory =  self.encoder.transformer(tokens)
        memory =  self.memory_to_dec(memory)

        num_masked = int(local_decoder_masking_ratio*self.hrrr_num_patches)

        hrrr_tokens = torch.zeros(batch, self.hrrr_num_patches, self.hrrr_decoder_dim).to(device)
        hrrr_tokens = self.decoder_pixel_to_emb(img_y)

        # JRD Test
        #print(pix_recon.shape)
        #print(hrrr_tokens.shape)

        # Must have at least one masked patch
        if num_masked == 0:
            num_masked = 10
            
        # At this point we treat it as fully masked
        if num_masked >= self.hrrr_num_patches - 10:

            rand_indices = torch.rand(batch, self.hrrr_num_patches, device=device)
            rand_indices = rand_indices.argsort(dim = -1)
            masked_indices, _ = rand_indices[:,:], None
            masked_patches = hrrr_patches[batch_range,masked_indices]

            hrrr_decoded_tokens = hrrr_tokens
            hrrr_decoded_tokens = self.enc_to_dec(hrrr_decoded_tokens)

            mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = self.hrrr_num_patches)
            mask_tokens = mask_tokens + self.enc_to_dec_pos(self.pos_embedding[0,masked_indices])

            decoder_tokens = torch.zeros(batch,self.hrrr_num_patches,self.decoder_dim, device=device)
            decoder_tokens[batch_range,masked_indices] = mask_tokens
            decoded_tokens = self.decoder(decoder_tokens, self.enc_to_dec2(memory)) + 0*hrrr_decoded_tokens

            mask_tokens = decoded_tokens[batch_range,masked_indices]
            pred_pixel_values = self.to_pixels(mask_tokens)

            #return pred_pixel_values[:,0,...], hrrr_patches, None
            #return pred_pixel_values, masked_patches, masked_indices
            return pred_pixel_values, masked_patches, masked_indices, None, None


        rand_indices = torch.rand(batch, self.hrrr_num_patches, device = device)
        rand_indices=rand_indices.argsort(dim = -1)

        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]
        masked_patches = hrrr_patches[batch_range, masked_indices]

        # get the unmasked tokens to be encoded
        hrrr_tokens = hrrr_tokens[batch_range, unmasked_indices]
        # get the patches to be masked for the final reconstruction loss
        masked_patches = hrrr_patches[batch_range, masked_indices]
        # attend with vision transformer
        #hrrr_decoded_tokens = self.hrrr_decoder.transformer(hrrr_tokens, torch.zeros_like(memory,device=memory.device)) # JRD!!!!
        hrrr_decoded_tokens = hrrr_tokens
        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        hrrr_decoded_tokens = self.enc_to_dec(hrrr_decoded_tokens)
        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = hrrr_decoded_tokens +\
            self.enc_to_dec_pos(self.pos_embedding[0,unmasked_indices])

        pix_recon = self.to_pixels(unmasked_decoder_tokens) # JRD

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens +\
            self.enc_to_dec_pos(self.pos_embedding[0,masked_indices])

        decoder_tokens = torch.zeros(batch, self.hrrr_num_patches, self.decoder_dim, device=device)
        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens, self.enc_to_dec2(memory))
        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        return pred_pixel_values, masked_patches, masked_indices, pix_recon, img_y[batch_range, unmasked_indices]

    def reconstruct(self, img, img_y, two_hours=False, decoder_masking_ratio=1):

        with torch.no_grad():
            pred_pixel_values, masked_patches, masked_indices = self.forward(img,img_y,two_hours=two_hours, decoder_masking_ratio=decoder_masking_ratio)

        return img_y, masked_indices, pred_pixel_values 


