"""TrackNet recovery using the same preprocessing as detect.py."""
import json
import time
import cv2
import numpy as np


class CachedHeavyDetector:
    """Fast replay of saved TrackNet detections; no neural inference occurs."""
    def __init__(self, path):
        with open(path,encoding='utf8') as f:
            data=json.load(f)
        self.fps=data['fps']
        self.points={round(o['timestamp']*self.fps):(np.array([o['position']['value']['x'],o['position']['value']['y']]),o['position'].get('confidence',1)) for o in data['ballObservations']}
        self.calls=0

    def detect(self,timestamp, center=None):
        # Saved full-frame detections cannot simulate inference on a crop.
        self.calls+=1
        value=self.points.get(round(timestamp*self.fps))
        return (value[0],value[1]) if value else None


class HeavyDetector:
    def __init__(self, video, end, background_samples=24):
        began = time.perf_counter()
        import torch
        from detect import TrackNet, ROOT

        torch.set_num_threads(4)
        cap = cv2.VideoCapture(str(video))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok or cap.get(cv2.CAP_PROP_POS_MSEC)/1000 >= end:
                break
            frames.append(cv2.resize(frame, (512,288)))
        cap.release()
        if not frames:
            raise ValueError('No frames for recovery detector')
        self.frames = np.asarray(frames,dtype=np.uint8)
        del frames
        indices = np.linspace(0,len(self.frames)-1,min(background_samples,len(self.frames)),dtype=int) if background_samples else np.arange(len(self.frames))
        self.background = np.median(self.frames[indices],axis=0).astype(np.float32)/255
        self.background_sample_count = len(indices)
        from mmengine.logging.history_buffer import HistoryBuffer
        with torch.serialization.safe_globals([HistoryBuffer,np.ndarray,np.dtype,(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),(np._core.multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),type(np.dtype('float64')),type(np.dtype('int64')),getattr]):
            state = torch.load(ROOT/'balltrack_best.pth',map_location='cpu',weights_only=True)
        state = state.get('state_dict',state)
        self.model = TrackNet(in_dim=15,out_dim=4,last_only=True).eval()
        self.model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()})
        self.calls = 0
        self.crop_calls = 0
        self.full_frame_calls = 0
        self.startup_seconds = time.perf_counter()-began

    def detect(self,timestamp, center=None):
        import torch

        i = round(timestamp*self.fps)
        if not 0 <= i < len(self.frames):
            return None
        left, top, width, height = 0, 0, 512, 288
        if center is not None:
            width, height = 256, 144
            cx, cy = np.asarray(center)*[512,288]
            left = int(np.clip(round(cx-width/2), 0, 512-width))
            top = int(np.clip(round(cy-height/2), 0, 288-height))
            self.crop_calls += 1
        else:
            self.full_frame_calls += 1
        # A shared crop preserves temporal alignment and the original ball scale.
        region = np.s_[top:top+height,left:left+width]
        sequence = [self.background[region],*[self.frames[max(0,j)][region].astype(np.float32)/255 for j in range(i-3,i+1)]]
        tensor = torch.from_numpy(np.stack(sequence).transpose(0,3,1,2).reshape(1,15,height,width))
        with torch.inference_mode():
            heatmap = self.model(frames=tensor)[0][0,0].numpy()
        self.calls += 1
        count, _, _, centers = cv2.connectedComponentsWithStats((heatmap>=.5).astype(np.uint8))
        if count != 2:
            return None
        # Components clipped by a crop boundary are unreliable; let recovery retry.
        point = centers[1]
        if center is not None and (point[0] < 4 or point[1] < 4 or point[0] >= width-4 or point[1] >= height-4):
            return None
        return (point+[left,top])/[512,288], float(heatmap.max())
