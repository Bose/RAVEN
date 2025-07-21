import numpy as np
import torch

ZERO_EMBEDDING_FOLDER = "./data/pretrained_zeros"

def augment_visual(face_embed, visual_encoder):
    random_mask = np.random.choice([time_mask, duplicate_frames, zero_frames])
    if random_mask == zero_frames:
        face_embed = random_mask(face_embed, visual_encoder)
    else:
        face_embed = random_mask(face_embed)
    
    return face_embed, random_mask.__name__


def time_mask(face_embed, duration=5, max_mask_duration=0.4, fps=25):
    """
    replace n consecutive frames with the mean face embeddings of the video
    
    randomly mask every second up to maximum seconds
    """
    assert face_embed.shape[0] == duration*fps, f"face embeddings and duration mismatch, face_embed shape is {face_embed.shape[0]} "
    avg_frame = torch.mean(face_embed, axis=0)
    
    indices = generate_indices(duration, max_mask_duration, fps)
    
    face_embed[indices] = avg_frame
    
    return face_embed


def duplicate_frames(face_embed, duration=5, max_duplicate_duration=0.4, fps=25):
    """
    duplicate n consecutive frames with the previous face embeddings of the video
    
    randomly duplicate every second up to maximum seconds
    """
    
    indices = generate_indices(duration, max_duplicate_duration, fps)
    
    prev_indices = indices - 1
    prev_indices[prev_indices < 0] = 0  
    face_embed[indices] = face_embed[prev_indices]
    
    return face_embed


def zero_frames(face_embed, visual_encoder, duration=5, max_zero_duration=0.4, fps=25):
    """
    replace n consecutive frames with embeddings that are obtained from zero input frames
    
    randomly replace with zero-fed embeddings every second up to maximum seconds
    """
    
    zero_embed_fp = f"{ZERO_EMBEDDING_FOLDER}/{visual_encoder}_zero_oneframe.npy"
    zero_frames_embed = torch.Tensor(np.load(zero_embed_fp))
    
    # Generate the indices where zero embeddings will be applied
    indices = generate_indices(duration, max_zero_duration, fps)
    
    # Replace the frames at the generated indices with the zero embeddings
    face_embed[indices] = zero_frames_embed
    
    return face_embed

def generate_indices(duration, max_duration, fps):
    """
    generate the indices per second on where to mask
    """
    mask_durations = torch.randint(1, int(max_duration * fps) + 1, (duration,))
    start_frames = torch.randint(0, fps - mask_durations.max() + 1, (duration,)) + torch.arange(duration) * fps
    expanded_indices = start_frames[:, None] + torch.arange(mask_durations.max())
    mask = expanded_indices < (start_frames[:, None] + mask_durations[:, None])
    return expanded_indices[mask]



    





        
    