"""Refine saved person tracks using RTMPose; retain IDs/timestamps and export comparisons."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from rtm_pose import RTMPose
from player_track import ROOT
from compare_pose_crops import panel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('video',type=Path);p.add_argument('tracks',type=Path)
    p.add_argument('--weights',type=Path,default=ROOT/'output/models/rtmpose-m.onnx')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--times',type=float,nargs='+')
    p.add_argument('--track-ids',type=int,nargs='+')
    p.add_argument('--comparison-dir',type=Path)
    p.add_argument('--flip-test', action='store_true', help='Average original and mirrored pose predictions')
    args=p.parse_args()
    data=json.loads(args.tracks.read_text(encoding='utf-8'))
    if data.get('format')!='pongmetrics-player-tracks' or data.get('version')!=1: raise ValueError('Unsupported tracks')
    if data['video']['name']!=args.video.name or data['video']['size']!=args.video.stat().st_size: raise ValueError('Video mismatch')
    rows={round(r['timestamp']*data['fps']):r for r in data['frames']}
    requested={round(t*data['fps']) for t in args.times} if args.times else set(rows)
    if not requested or not requested.issubset(rows): raise ValueError('Requested frames must have saved boxes')
    started=time.perf_counter();model=RTMPose(args.weights, flip_test=args.flip_test);startup=time.perf_counter()-started
    cv2.setNumThreads(1)
    cap=cv2.VideoCapture(str(args.video));output=[];calls=0;elapsed=0.;i=0
    if args.comparison_dir: args.comparison_dir.mkdir(parents=True,exist_ok=True)
    try:
        while i<=max(requested):
            ok,image=cap.read()
            if not ok: raise ValueError('Source ended before requested frames')
            if i in requested:
                row=copy.deepcopy(rows[i]);row['people']=[];tiles=[]
                h,w=image.shape[:2]
                for original in rows[i]['people']:
                    if args.track_ids and original['trackId'] not in args.track_ids: continue
                    person=copy.deepcopy(original);b=person['box']
                    box=[b['x']*w,b['y']*h,(b['x']+b['width'])*w,(b['y']+b['height'])*h]
                    tick=time.perf_counter();person['landmarks']=model.predict(image,box);elapsed+=time.perf_counter()-tick;calls+=1
                    row['people'].append(person)
                    if args.comparison_dir and original.get('landmarks'):
                        pad=np.array([b['width']*w,b['height']*h])*.25
                        left,top=np.maximum(0,np.floor(np.array(box[:2])-pad)).astype(int)
                        right,bottom=np.minimum([w,h],np.ceil(np.array(box[2:])+pad)).astype(int)
                        region=(left,top,right,bottom)
                        tiles.append(np.hstack([panel(image,original,region,f'{i/data["fps"]:.2f}s #{person["trackId"]} YOLO full frame'),panel(image,person,region,'RTMPose-m source player region')]))
                output.append(row)
                if tiles: cv2.imwrite(str(args.comparison_dir/f'comparison-{i}.jpg'),np.vstack(tiles))
                if len(output)%100==0: print(f'{len(output)} frames refined',flush=True)
            i+=1
    finally: cap.release()
    parent=data.get('model')
    data['frames']=output
    width, height = model.input_size
    data['model']={'name':f'RTMPose COCO-17 {height}x{width}','sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                   'runtime':'onnxruntime-cpu','inputSize':[width,height],'personBoxPadding':1.25,'parentDetector':parent,
                   'flipTest':args.flip_test,
                   'confidenceMeaning':'clipped SimCC peak strength; not calibrated probability or visibility; rawScore retained'}
    data['summary']={'frames':len(output),'poseCalls':calls,'inferenceCalls':calls * (2 if args.flip_test else 1),
                     'poseSeconds':elapsed,'startupSeconds':startup,'totalSeconds':time.perf_counter()-started}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(data['summary']),flush=True)


if __name__=='__main__': main()
