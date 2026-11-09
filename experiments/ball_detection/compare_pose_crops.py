"""Compare saved full-frame pose with source-resolution player crops on a few frames."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from player_track import BONES, KEYPOINT_NAMES, ROOT, normalized_landmarks


def lift_landmarks(points, left, top, width, height):
    points=np.array(points,dtype=float,copy=True)
    missing=(points[:,:2]==0).all(axis=1)
    points[:,:2]+=[left,top]
    points[missing,:2]=np.nan
    return normalized_landmarks(points,width,height)


def iou(a,b):
    overlap=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    intersection=float(np.prod(overlap))
    return intersection/max(1e-9,float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2]))-intersection)


def panel(image, person, region, title):
    left,top,right,bottom=region
    crop=image[top:bottom,left:right]
    scale=min(400/crop.shape[1],400/crop.shape[0])
    resized=cv2.resize(crop,None,fx=scale,fy=scale)
    canvas=np.zeros((448,400,3),dtype=np.uint8)
    canvas[48:48+resized.shape[0],:resized.shape[1]]=resized
    joints=person['landmarks']
    h,w=image.shape[:2]
    def point(index):
        p=joints[index]['position']
        return (round((p['value']['x']*w-left)*scale),48+round((p['value']['y']*h-top)*scale)) if p and p['confidence']>=.35 else None
    for a,b in BONES:
        p,q=point(a),point(b)
        if p and q: cv2.line(canvas,p,q,(255,190,0),1)
    for index in range(5,17):
        p=point(index)
        if p:
            cv2.circle(canvas,p,3,(0,255,255),1)
            cv2.putText(canvas,str(index),p,cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255),1)
    cv2.putText(canvas,title,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
    scores=[joints[i]['position']['confidence'] if joints[i]['position'] else 0 for i in [9,10]]
    cv2.putText(canvas,f'wrist scores L {scores[0]:.2f} R {scores[1]:.2f}',(5,38),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    return canvas


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path)
    parser.add_argument('baseline',type=Path)
    parser.add_argument('--times',type=float,nargs='+',required=True)
    parser.add_argument('--track-ids',type=int,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.baseline.read_text(encoding='utf-8'))
    if args.video.name!=data['video']['name'] or args.video.stat().st_size!=data['video']['size']:
        raise ValueError('Video must match saved pose observations')
    weights=ROOT/'output/models/yolo11n-pose.pt'
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=data['model']['sha256']:
        raise ValueError('Use the same weights as the baseline')
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    model=YOLO(str(weights))
    requested={round(t*data['fps']) for t in args.times}
    rows={round(r['timestamp']*data['fps']):r for r in data['frames']}
    if not requested.issubset(rows): raise ValueError('Requested times must have saved baseline poses')
    args.output.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(args.video))
    output_rows=[]
    comparisons=[]
    index=0
    inference=0.
    calls=0
    try:
        while index<=max(requested):
            ok,image=cap.read()
            if not ok: raise ValueError('Video ended before requested frame')
            if index in requested:
                h,w=image.shape[:2]
                output_row=copy.deepcopy(rows[index])
                output_row['people']=[]
                tiles=[]
                for person in rows[index]['people']:
                    if person['trackId'] not in args.track_ids: continue
                    box=person['box']
                    target=np.array([box['x']*w,box['y']*h,(box['x']+box['width'])*w,(box['y']+box['height'])*h])
                    # 25% padding per side retains hands extending beyond a box.
                    pad=np.array([box['width']*w,box['height']*h])*.25
                    left,top=np.maximum(0,np.floor(target[:2]-pad)).astype(int)
                    right,bottom=np.minimum([w,h],np.ceil(target[2:]+pad)).astype(int)
                    tick=time.perf_counter()
                    result=model.predict(image[top:bottom,left:right],imgsz=640,conf=.1,device='cpu',verbose=False)[0]
                    inference+=time.perf_counter()-tick
                    calls+=1
                    refined=copy.deepcopy(person)
                    refined['landmarks']=[{'name':name,'position':None} for name in KEYPOINT_NAMES]
                    best_iou=0.
                    if result.boxes is not None and result.keypoints is not None:
                        boxes=result.boxes.xyxy.cpu().numpy()+[left,top,left,top]
                        matches=[iou(box,target) for box in boxes]
                        if matches:
                            best=int(np.argmax(matches))
                            best_iou=matches[best]
                            if best_iou>=.25:
                                refined['landmarks']=lift_landmarks(result.keypoints.data[best].cpu().numpy(),left,top,w,h)
                    output_row['people'].append(refined)
                    comparisons.append(dict(frame=index,trackId=person['trackId'],crop=[int(left),int(top),int(right),int(bottom)],matched=best_iou>=.25,matchIoU=best_iou,
                                            baselineLandmarks=person['landmarks'],cropLandmarks=refined['landmarks']))
                    region=(left,top,right,bottom)
                    tiles.append(np.hstack([panel(image,person,region,f'{index/data["fps"]:.2f}s #{person["trackId"]} full-frame input'),
                                            panel(image,refined,region,'Same model / original-player crop')]))
                if tiles: cv2.imwrite(str(args.output/f'comparison-{index}.jpg'),np.vstack(tiles))
                output_rows.append(output_row)
            index+=1
    finally:
        cap.release()
    output=copy.deepcopy(data)
    output['frames']=output_rows
    output['model'].update(poseInput='source-player-crop',cropPaddingPerSide=.25,minimumMatchIoU=.25)
    output['summary']={'frames':len(output_rows),'cropCalls':calls,'cropInferenceSeconds':inference,'matchedCrops':sum(c['matched'] for c in comparisons)}
    (args.output/'player-tracks.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    (args.output/'comparison.json').write_text(json.dumps(comparisons,indent=2),encoding='utf-8')
    print(json.dumps(output['summary']))


if __name__=='__main__': main()
