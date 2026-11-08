"""Decode every official rollout video and check it against executed steps."""
import argparse,json
from pathlib import Path
import cv2


def validate(root,out):
    root=Path(root).resolve();rows=[]
    for records_path in sorted(root.rglob('episodes.jsonl')):
        for line in records_path.read_text().splitlines():
            r=json.loads(line);videos=list(records_path.parent.glob(f'*--episode={r["episode"]+1}--success=*--task=xpnpcountertosink.mp4'))
            if len(videos)!=1:raise RuntimeError(f'Ambiguous/missing video for {records_path}, {r["episode"]}')
            path=videos[0];cap=cv2.VideoCapture(str(path));frames=0;size=None
            while True:
                ok,frame=cap.read()
                if not ok:break
                frames+=1;size=[int(frame.shape[1]),int(frame.shape[0])]
            cap.release()
            row=dict(video_path=str(path),episode_record=str(records_path),episode=r['episode'],expected_frames=r['steps'],decoded_frames=frames,encoded_dimensions=size,verified=frames==r['steps'] and frames>0)
            rows.append(row)
            if not row['verified']:raise RuntimeError('Incomplete video: '+str(path))
    Path(out).write_text(json.dumps(dict(videos=len(rows),all_verified=bool(rows) and all(r['verified'] for r in rows),records=rows),indent=2))
    print('Fully decoded',len(rows),'videos,',sum(r['decoded_frames'] for r in rows),'frames')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();validate(a.root,a.out)
