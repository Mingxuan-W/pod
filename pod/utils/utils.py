from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from typing import Union
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

da_image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
da_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
da_model.to('cuda')
def get_depth(img: Union[torch.tensor,np.ndarray]):
    assert img.shape[2] == 3
    if isinstance(img,torch.Tensor):
        img = img.cpu().numpy()
    image = Image.fromarray(img)

    # prepare image for the model
    inputs = da_image_processor(images=image, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].cuda()

    with torch.no_grad():
        outputs = da_model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    return prediction.squeeze()

hand_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
hand_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
hand_model.to('cuda')
def get_hand_mask(img: Union[torch.tensor,np.ndarray]):
    assert img.shape[2] == 3
    if isinstance(img,torch.Tensor):
        img = img.cpu().numpy()
    image = Image.fromarray(img)

    # prepare image for the model
    inputs = hand_processor(images=image, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].cuda()

    with torch.no_grad():
        outputs = hand_model(**inputs)

    # Perform post-processing to get panoptic segmentation map
    seg_ids = hand_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    hand_mask = (seg_ids == hand_model.config.label2id['person']).float()
    return hand_mask



def build_parent_mapping(tree_structure):
    parent_map = {}
    for parent, children in tree_structure.items():
        for child in children:
            parent_map[child] = parent
    return parent_map



def scatter_images(tsne_features, images, ax, fig, original_embeddings):
    # tsne is Nx2
    # images is Nx3xhxw
    # first resize the images to be easier to draw
    import cv2
    from matplotlib.backend_bases import MouseButton
    from scipy.spatial import geometric_slerp
    from sklearn.manifold import TSNE
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    
    resized = []
    for i in range(images.shape[0]):
        resized.append(cv2.resize(images[i, ...], (64, 64)))

    images = np.stack(resized, axis=0)
    artists = []
    select1, select2 = None, None
    default_zoom = .9

    def artist_cb(art, event):
        nonlocal select1, select2, artists
        if art.contains(event)[0]:
            if event.button == MouseButton.LEFT and select1 is None:
                art.get_children()[0].set_zoom(default_zoom * 2)
                select1 = art.index
            elif event.button == MouseButton.RIGHT and select2 is None:
                art.get_children()[0].set_zoom(default_zoom * 2)
                select2 = art.index
            fig.canvas.draw()
            return True, {}
        else:
            return False, {}

    def on_press(event):
        nonlocal select1, select2, artists, artist_cb
        if event.key == 'i':
            artists = []
            ax.clear()
            vec1 = original_embeddings[select1, :].astype(np.float64)
            vec2 = original_embeddings[select2, :].astype(np.float64)
            vec1 /= np.linalg.norm(vec1)
            vec2 /= np.linalg.norm(vec2)
            interps = geometric_slerp(vec1, vec2, np.linspace(0, 1, 40)).astype(np.float32)
            new_embeddings = np.concatenate((original_embeddings, interps), axis=0)
            tsne = TSNE(metric='cosine', init='pca')
            vis_vectors = tsne.fit(new_embeddings).embedding_
            # plot the originals with the axes objects
            for i in range(original_embeddings.shape[0]):
                im = OffsetImage(images[i, ...], zoom=default_zoom)
                ab = AnnotationBbox(im, (vis_vectors[i, 0], vis_vectors[i, 1]), xycoords='data',
                                    frameon=False, picker=artist_cb)
                ab.index = i
                ax.add_artist(ab)
                artists.append(ab)
            # plot the interpolation data
            ax.scatter(vis_vectors[original_embeddings.shape[0]:, 0],
                        vis_vectors[original_embeddings.shape[0]:, 1], marker='.')
            ax.update_datalim(vis_vectors)
            ax.autoscale()
            fig.canvas.draw()
            select1, select2 = None, None
        elif event.key == 'r':
            for a in artists:
                a.get_children()[0].set_zoom(default_zoom)
            select1, select2 = None, None
            fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_press)
    for i in range(tsne_features.shape[0]):
        im = OffsetImage(images[i, ...], zoom=default_zoom)
        ab = AnnotationBbox(im, (tsne_features[i, 0], tsne_features[i, 1]), xycoords='data',
                            frameon=False, picker=artist_cb)
        ab.index = i
        ax.add_artist(ab)
        artists.append(ab)
    ax.update_datalim(tsne_features)
    ax.autoscale()

def process_data(video_data):
    n_videos = len(video_data)
    n_frames_per_video = [len(video_data[i]['rgb']) for i in range(n_videos)]
    idx_map = [(i,j) for i in range(n_videos) for j in range(n_frames_per_video[i])]
    rgb_frames = torch.cat([video_data[i]['rgb'] for i in range(n_videos)],dim=0)
    
    return rgb_frames, n_videos, n_frames_per_video, idx_map

def correct_t(R, T):
    # R: [batch, n, 3, 3] - Rotation matrices
    # T: [batch, n, 3] - Translation vectors
    
    # Ensure R and T are tensors
    if not isinstance(R, torch.Tensor) or not isinstance(T, torch.Tensor):
        raise TypeError("Both R and T should be torch tensors")
    
    # Perform the matrix-vector multiplication: -R @ T + T
    # Expand T to match the batch and matrix dimensions
    T_expanded = T.unsqueeze(-1)  # [batch, n, 3] -> [batch, n, 3, 1]
    
    # Apply the transformation: -R @ T_expanded + T
    transformed_T = -torch.matmul(R, T_expanded).squeeze(-1) + T  # Squeeze to get back to [batch, n, 3]

    return transformed_T


hand_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
hand_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
hand_model.to('cuda')
def get_hand_mask(img: Union[torch.tensor,np.ndarray]):
    assert img.shape[2] == 3
    if isinstance(img,torch.Tensor):
        img = img.cpu().numpy()
    image = Image.fromarray(img)

    # prepare image for the model
    inputs = hand_processor(images=image, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].cuda()

    with torch.no_grad():
        outputs = hand_model(**inputs)

    # Perform post-processing to get panoptic segmentation map
    seg_ids = hand_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    hand_mask = (seg_ids == hand_model.config.label2id['person']).float()
    return hand_mask