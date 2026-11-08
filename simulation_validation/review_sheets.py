"""Create labeled video samples for each failed physical rollout.

Samples aid review alongside full state/action histories. They are not an
occlusion metric, a dense video annotation, or an automatic causal classifier.
"""
import argparse,json
from pathlib import Path
import cv2
import numpy as np


def generate(analysis,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[r for r in json.loads(Path(analysis).read_text())['episodes'] if not r['success']]
    manifest=[]
    for group_index in range(0,len(rows),5):
        group=rows[group_index:group_index+5];canvas=np.full((len(group)*164,6*240,3),240,np.uint8);entries=[]
        for y,row in enumerate(group):
            cap=cv2.VideoCapture(row['video_path']);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count<=0:raise RuntimeError('Undecodable video: '+row['video_path'])
            indices=[0,min(120,count-1),min(row['closest_step'],count-1),min(row['max_rise_step'],count-1),min(480,count-1),count-1]
            for x,index in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
                if not ok:raise RuntimeError(f'Cannot decode {row["case_id"]} frame {index}')
                frame=cv2.resize(frame,(240,135));canvas[y*164+29:y*164+164,x*240:(x+1)*240]=frame
                cv2.putText(canvas,f'{row["case_id"]} f{index}',(x*240+3,y*164+20),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),1)
            cap.release();entries.append(dict(case_id=row['case_id'],video_path=row['video_path'],reviewed_frame_steps=indices))
        path=out/f'failures-{group_index//5:03d}.jpg';cv2.imwrite(str(path),canvas)
        manifest.append(dict(sheet=str(path.resolve()),cases=entries))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--out',required=True);a=p.parse_args();generate(a.analysis,a.out)
