"""TrackNet recovery using the same preprocessing as detect.py."""
import cv2
import numpy as np
import torch
from detect import TrackNet, ROOT


class HeavyDetector:
    def __init__(self, video, end):
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
        self.background = np.median(np.stack(frames),axis=0).astype(np.float32)/255
        self.frames = np.asarray(frames,dtype=np.float32)/255
        from mmengine.logging.history_buffer import HistoryBuffer
        with torch.serialization.safe_globals([HistoryBuffer,np.ndarray,np.dtype,(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),(np._core.multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),type(np.dtype('float64')),type(np.dtype('int64')),getattr]):
            state = torch.load(ROOT/'balltrack_best.pth',map_location='cpu',weights_only=True)
        state = state.get('state_dict',state)
        self.model = TrackNet(in_dim=15,out_dim=4,last_only=True).eval()
        self.model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()})
        self.calls = 0

    def detect(self,timestamp):
        i = round(timestamp*self.fps)
        if not 0 <= i < len(self.frames):
            return None
        sequence = [self.background,*[self.frames[max(0,j)] for j in range(i-3,i+1)]]
        tensor = torch.from_numpy(np.stack(sequence).transpose(0,3,1,2).reshape(1,15,288,512))
        with torch.inference_mode():
            heatmap = self.model(frames=tensor)[0][0,0].numpy()
        self.calls += 1
        count, _, _, centers = cv2.connectedComponentsWithStats((heatmap>=.5).astype(np.uint8))
        if count != 2:
            return None
        return centers[1]/[512,288], float(heatmap.max())
