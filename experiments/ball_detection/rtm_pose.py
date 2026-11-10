"""CPU RTMPose on source-image person boxes, using standard SimCC decoding.

Pre/postprocessing follows OpenMMLab's RTMPose ONNX example (Apache-2.0):
https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose/examples/onnxruntime
No smoothing, interpolation, or visibility classification is implied.
"""
from pathlib import Path
import cv2
import numpy as np

from player_track import normalized_landmarks


def prepare_box(box, input_size=(192,256), padding=1.25):
    box=np.asarray(box,dtype=np.float32)
    if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[2:]<=box[:2]):
        raise ValueError('Expected a finite positive xyxy box')
    center=(box[:2]+box[2:])/2
    scale=(box[2:]-box[:2])*padding
    ratio=input_size[0]/input_size[1]
    if scale[0]>scale[1]*ratio: scale[1]=scale[0]/ratio
    else: scale[0]=scale[1]*ratio
    matrix=np.array([[input_size[0]/scale[0],0,input_size[0]*(.5-center[0]/scale[0])],
                     [0,input_size[1]/scale[1],input_size[1]*(.5-center[1]/scale[1])]],dtype=np.float32)
    return center,scale,matrix


def decode_simcc(x,y,center,scale,input_size=(192,256)):
    if x.shape[:2]!=(1,17) or y.shape[:2]!=(1,17): raise ValueError('Expected 17-joint SimCC output')
    locations=np.stack([x.argmax(axis=-1),y.argmax(axis=-1)],axis=-1)[0]/2
    scores=np.minimum(x.max(axis=-1),y.max(axis=-1))[0]
    points=locations/np.asarray(input_size)*scale+center-scale/2
    points[scores<=0]=np.nan
    return points,scores


class RTMPose:
    def __init__(self,weights:Path):
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(weights),sess_options=options,providers=['CPUExecutionProvider'])
        item=self.session.get_inputs()[0]
        self.input_name=item.name
        self.input_size=(int(item.shape[3]),int(item.shape[2]))
        if self.input_size!=(192,256): raise ValueError('This adapter expects the RTMPose-m 256x192 export')

    def predict(self,image,box):
        height,width=image.shape[:2]
        center,scale,matrix=prepare_box(box,self.input_size)
        crop=cv2.warpAffine(image,matrix,self.input_size,flags=cv2.INTER_LINEAR)
        # Official export expects BGR with these channel-wise normalization values.
        tensor=((crop.astype(np.float32)-np.array([123.675,116.28,103.53],np.float32))/np.array([58.395,57.12,57.375],np.float32)).transpose(2,0,1)[None]
        outputs=self.session.run(None,{self.input_name:tensor})
        points,scores=decode_simcc(*outputs,center,scale,self.input_size)
        # SimCC strengths are not visibility probabilities or YOLO-equivalent scores.
        joints=normalized_landmarks(np.column_stack([points,np.clip(scores,0,1)]),width,height)
        for joint,score in zip(joints,scores): joint['rawScore']=float(score) if np.isfinite(score) else None
        return joints
