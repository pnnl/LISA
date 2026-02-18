import einops
import torch
import os.path as Path
import matplotlib.pyplot as plt
import os

def reconstruct_image_reduced(mask_indices, ground_truth, prediction):

    mask_indices = list(mask_indices)

    image_size = (2 ,670, 260)
    blank_image  = torch.zeros(image_size)
    gt_patches   = einops.rearrange(blank_image,'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1 = 10, p2 = 10)
    pred_patches = einops.rearrange(blank_image,'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1 = 10, p2 = 10)

    patch_index = 0
    for patch_id in range(0,gt_patches.shape[0]):

        if patch_id in mask_indices:
            gt_patches[patch_id,:]   = ground_truth[patch_index]
            pred_patches[patch_id,:] = prediction[patch_index]

            patch_index += 1

    gt_patches   = einops.rearrange(gt_patches,' (h w) (p1 p2 c) -> c (h p1) (w p2)', p1 = 10, p2 = 10, h = 67, w=26)
    pred_patches = einops.rearrange(pred_patches,' (h w) (p1 p2 c) -> c (h p1) (w p2)', p1 = 10, p2 = 10, h = 67, w=26)

    return gt_patches, pred_patches

def plot_reconstruction(gt,pred,path,epoch,step):
        
    if not Path.exists(path):
        os.makedirs(path)

    f, ax = plt.subplots(2,2,figsize=(12,12))

    ax[0,0].imshow(gt[0])
    ax[1,0].imshow(gt[1])
    ax[0,1].imshow(pred[0])
    ax[1,1].imshow(pred[1])

    ax[0,0].set_title('Ground Truth, U')
    ax[1,0].set_title('Ground Truth, V')
    ax[0,1].set_title('Prediction, U')
    ax[1,1].set_title('Prediction, V')
    plt.tight_layout()
    plt.savefig(f'{path}/comparison_{epoch}_{step}.png')

    return





def reconstruct_image( patches, model_input, masked_indices=None, pred_pixel_values=None, patch_size=8):
    """
    Reconstructs the image given patches. Can also reconstruct the masked image as well as the predicted image.
    To reconstruct the raw image from the patches, set masked_indices=None and pred_pixel_values=None. To reconstruct
    the masked image, set masked_indices= the masked_indices tensor created in the `forward` call. To reconstruct the
    predicted image, set masked_indices and pred_pixel_values = to their respective tensors created in the `forward` call.

    ARGS:
        patches (torch.Tensor): The raw patches (pre-patch embedding) generated for the given model input. Shape is
            (batch_size x num_patches x patch_size^2 * channels)
        model_input (torch.Tensor): The input images to the given model (batch_size x channels x height x width)
        mean (list[float]): An array representing the per-channel mean of the dataset used to
            denormalize the input and predicted pixels. (1 x channels)
        std (list[float]): An array representing the per-channel std of the dataset used to
            denormalize the input and predicted pixels. (1 x channels)
        masked_indices (torch.Tensor): The patch indices that are masked (batch_size x masking_ratio * num_patches)
        pred_pixel_values (torch.Tensor): The predicted pixel values for the patches that are masked (batch_size x masking_ratio * num_patches x patch_size^2 * channels)

    RETURN:
        reconstructed_image (torch.Tensor): Tensor containing the reconstructed image (batch_size x channels x height x width)
    """
    patches = patches.cpu()

    masked_indices_in = masked_indices is not None
    predicted_pixels_in = pred_pixel_values is not None

    if masked_indices_in:
        masked_indices = masked_indices.cpu()

    if predicted_pixels_in:
        pred_pixel_values = pred_pixel_values.cpu()

    patch_width = patch_height = patch_size
    reconstructed_image = patches.clone()

    if masked_indices_in or predicted_pixels_in:
        for i in range(reconstructed_image.shape[0]):
            if masked_indices_in and predicted_pixels_in:
                reconstructed_image[i, masked_indices[i].cpu()] = pred_pixel_values[i, :].cpu().float()
            elif masked_indices_in:
                reconstructed_image[i, masked_indices[i].cpu()] = 0

    reconstructed_image = einops.rearrange(reconstructed_image,'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', w=int(model_input.shape[3] / patch_width),
                                h=int(model_input.shape[2] / patch_height), c=model_input.shape[1],
                                p1=patch_height, p2=patch_width)

    reconstructed_image = einops.rearrange(reconstructed_image,'a b c d -> a d c b')

    return reconstructed_image