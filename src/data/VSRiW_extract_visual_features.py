
import os
import torch
import numpy as np
import pandas as pd
import sys
import src.config as config
sys.path.append(config.VSRIW_PATH)
from pipelines.model import AVSR
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, get_worker_info
import torchvision
from pathlib import Path
from mediapipe.python.solutions.face_detection import FaceDetection, FaceKeyPoint
from pipelines.detectors.mediapipe.video_process import VideoProcess
from pipelines.data.transforms import VideoTransform

DATA_FOLDER_PATH = config.DATA_FOLDER_PATH
class VideoDataset(Dataset):
    def __init__(
            self, 
            data_split_path,
            data_path,
            split,
            face_track: bool = True, 
            ) -> None:
        super().__init__()
        self.df = pd.read_csv(data_split_path)
        self.data_path = data_path
        self.split = split
        self.file_list = pd.read_csv('./data/split.csv', header=None)[0].str.replace("/aac/", "/mp4/").str.replace(".m4a", ".mp4").tolist()
        
        self.video_processor = VideoProcess()
        self.video_transform = VideoTransform(speed_rate=1)
        self.face_track = face_track
        self.short_range_detector: FaceDetection = None
        self.long_range_detector: FaceDetection = None 

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        fp = os.path.join(self.data_path, self.file_list[index])
        video_tensor = torchvision.io.read_video(fp, pts_unit='sec')[0].numpy()
        video_tensor = random_5s_clip(video_tensor)
        try:
            landmarks = self._process_landmarks(video_tensor)
            video_tensor = self.video_processor(video_tensor, landmarks)
            video_tensor = self.video_transform(torch.tensor(video_tensor))
            return video_tensor, fp
        except:
            return None, fp
        
    def _initialize_detector(self):
        self.short_range_detector = FaceDetection(min_detection_confidence=0.5, model_selection=0)
        self.long_range_detector = FaceDetection(min_detection_confidence=0.5, model_selection=1)

    def _process_landmarks(self, video_tensor):
        if self.face_track:
            landmarks = self._detect(video_tensor, self.long_range_detector)
            if all(l is None for l in landmarks):
                landmarks = self._detect(video_tensor, self.short_range_detector)
                if all(l is None for l in landmarks):
                    UserWarning("Cannot detect any frames in the video")
            return landmarks
        else:
            return None
    
    def _detect(self, video_frames, detector):
        landmarks = []
        for frame in video_frames:
            results = detector.process(frame)
            if not results.detections:
                landmarks.append(None)
                continue
            face_points = []
            for idx, detected_faces in enumerate(results.detections):
                max_id, max_size = 0, 0
                bboxC = detected_faces.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                if bbox_size > max_size:
                    max_id, max_size = idx, bbox_size
                lmx = [
                    [int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(0).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(0).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(1).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(1).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(2).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(2).value].y * ih)],
                    [int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(3).value].x * iw),
                     int(detected_faces.location_data.relative_keypoints[FaceKeyPoint(3).value].y * ih)],
                    ]
                face_points.append(lmx)
            landmarks.append(np.array(face_points[max_id]))
        return landmarks


def video_dataset_worker_init(worker_id):
    info = get_worker_info()
    info.dataset._initialize_detector()
    
def random_5s_clip(video, frame_rate=25):

    T, H, W, C = video.shape
    target_frames = 5 * frame_rate 

    if T >= target_frames:
        
        return video[:target_frames, :, :, :]
    else:
        pad_frames = target_frames - T
        pad_array = np.zeros((pad_frames, H, W, C), dtype=video.dtype) 
        return np.concatenate((video, pad_array), axis=0)

class AVSRBatchedPreprocessing:
    def __init__(
            self,
            model_path,
            model_conf,
            data_split_path,
            split,
            data_path= True,
            batch_size= None,
            num_workers: int = 1,
            face_track: bool = True, 
            device = "cuda"
        ) -> None:
        super().__init__()
        self.split = split
        self.ds = VideoDataset(data_split_path, data_path, split=self.split, face_track=face_track)
        self.dataloader = DataLoader(self.ds, batch_size=batch_size, num_workers=num_workers, drop_last=False, worker_init_fn=video_dataset_worker_init)
        self.model = AVSR("video", model_path, model_conf, rnnlm=None, rnnlm_conf=None, penalty=0.0, ctc_weight=0.1, lm_weight=0.0, beam_size=40, device=device)
        self.device = device

    def extract_features(self):
        failed_text_path = os.path.join(config.DATA_FOLDER_PATH, f"failed_VSRiW_frontE_feat_{self.split}.txt")
        with open(failed_text_path, 'w') as failed_txt:
            for video_tensor, fp in tqdm(self.dataloader):
                if video_tensor is not None:
                    output_ft_path = fp.replace("/mp4/", "/vsriw/")
                    output_ft_path = output_ft_path.replace(".mp4", ".npy")
                    if os.path.exists(output_ft_path):
                        continue
                    if not os.path.exists(os.path.dirname(output_ft_path)):
                        os.makedirs(os.path.dirname(output_ft_path))
                    with torch.no_grad():
                        enc_feats: torch.Tensor = self.model.model.encode(video_tensor.to(self.device), True)
                    
                    features = enc_feats.detach().cpu().numpy()
                    np.save(output_ft_path, features)

                else:
                    print(f"Video {fp} preprocessing failed.")
                    failed_txt.write(fp)
    
    def test_zero(self):
        
        with torch.no_grad():

            video_tensor = torch.zeros(1, 1, 88, 88)
            video_embed = self.model.model.encode(video_tensor.to(self.device), True)
                
   
                
            np.save(f"./pretrained_zeros/VSRiW_zero_oneframe.npy", video_embed.detach().cpu().numpy())
            
            print(video_embed.shape)
            print(video_embed)
                

def main():
    model_conf = os.path.join(config.VSRIW_PATH, "benchmarks/GRID/models/model.json")
    model_path = os.path.join(config.VSRIW_PATH, "benchmarks/GRID/models/model.pth")
    data_split = "./data/split.csv"
    process = AVSRBatchedPreprocessing(
        model_path=model_path,
        model_conf=model_conf,
        data_split_path=data_split,
        data_path=DATA_FOLDER_PATH,
        split="val",
        face_track=True,
        num_workers=64
    )
    process.extract_features()


if __name__ == "__main__":
    os.environ["GLOG_minloglevel"] ="2"

    main()
    
    