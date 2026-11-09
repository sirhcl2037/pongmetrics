"""Render saved pose near candidate contacts; no inference or training."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from player_track import BONES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path)
    parser.add_argument('tracks',type=Path)
    parser.add_argument('--times',type=float,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.tracks.read_text(encoding='utf-8'))
    if args.video.name != data['video']['name'] or args.video.stat().st_size != data['video']['size']:
        raise ValueError('Tracks must match the source video')
    fps=data['fps']
    rows={round(r['timestamp']*fps):r for r in data['frames']}
    centers=[round(t*fps) for t in args.times]
    requested={i for center in centers for i in [center-1,center,center+1]}
    cap=cv2.VideoCapture(str(args.video))
    tiles={}
    records=[]
    index=0
    try:
        while index<=max(requested):
            ok,image=cap.read()
            if not ok: break
            if index in requested:
                h,w=image.shape[:2]
                for person in rows.get(index,{}).get('people',[]):
                    if person['playerId'] not in ('near-player','far-player'): continue
                    box=person['box']
                    # Constant aspect ratio with original pixels before enlargement.
                    cx,cy=(box['x']+box['width']/2)*w,(box['y']+box['height']/2)*h
                    size=max(box['width']*w,box['height']*h)*1.35
                    x0,y0=max(0,int(cx-size/2)),max(0,int(cy-size/2))
                    x1,y1=min(w,int(cx+size/2)),min(h,int(cy+size/2))
                    tile=np.zeros((400,400,3),dtype=np.uint8)
                    crop=image[y0:y1,x0:x1]
                    scale=min(400/crop.shape[1],400/crop.shape[0])
                    resized=cv2.resize(crop,None,fx=scale,fy=scale)
                    tile[:resized.shape[0],:resized.shape[1]]=resized
                    joints=person.get('landmarks',[])
                    if len(joints) != 17:
                        raise ValueError('Pose audit requires 17 saved COCO joints per person')
                    def point(j):
                        p=joints[j]['position']
                        return (round((p['value']['x']*w-x0)*scale),round((p['value']['y']*h-y0)*scale)) if p and p['confidence']>=.35 else None
                    for a,b in BONES:
                        p,q=point(a),point(b)
                        if p and q: cv2.line(tile,p,q,(255,190,0),1)
                    scores={}
                    for j in [5,6,7,8,9,10,11,12]:
                        p=point(j)
                        observation=joints[j]['position']
                        scores[joints[j]['name']]=observation['confidence'] if observation else None
                        if p:
                            cv2.circle(tile,p,3,(0,255,255),1)
                            cv2.putText(tile,str(j),p,cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
                    tile=cv2.copyMakeBorder(tile,48,0,0,0,cv2.BORDER_CONSTANT)
                    cv2.putText(tile,f"{index/fps:.2f}s f{index} {person['playerId']} #{person['trackId']}",(5,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
                    wrist=lambda j: '-' if scores[j] is None else f'{scores[j]:.2f}'
                    cv2.putText(tile,f"wrist scores L {wrist('left-wrist')} R {wrist('right-wrist')}",(5,38),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                    tiles[index,person['playerId']]=tile
                    records.append(dict(frame=index,timestamp=index/fps,trackId=person['trackId'],scores=scores))
            index+=1
    finally:
        cap.release()
    args.output.mkdir(parents=True,exist_ok=True)
    for center in centers:
        sheet=np.vstack([np.hstack([tiles.get((i,side),np.zeros((448,400,3),dtype=np.uint8)) for i in [center-1,center,center+1]]) for side in ['far-player','near-player']])
        cv2.imwrite(str(args.output/f'contact-{center}.jpg'),sheet)
    (args.output/'scores.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps({'contacts':len(centers),'playerFrames':len(records),'output':str(args.output)}))


if __name__=='__main__':
    main()
