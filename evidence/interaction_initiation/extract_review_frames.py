"""Extract original decoded MP4 frames for mask-feasibility inspection; no edits."""
import csv,json,hashlib
from pathlib import Path
import cv2,h5py

REPO=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
HOST=Path('/home/users/lemon/interaction-initiation-20261110')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    matches=list(csv.DictReader((OUT/'source_initiation_query_matches.csv').open()))
    selected=[r for r in matches if r['history_calipers_pass']=='True']
    # Review every successful descriptive history match, including agreeing controls.
    episodes={(r['task'],r['robot'],r['condition']):r for r in csv.DictReader((OUT/'episodes.csv').open())}
    index=[]
    for row in selected:
        for robot,column in [('ur5e','source_query_action'),('panda','target_query_action')]:
            episode=episodes[row['task'],robot,row['condition']];t=int(row[column]);video=Path(episode['video_path'])
            cap=cv2.VideoCapture(str(video));assert cap.isOpened()
            for _ in range(t+1):ok,frame=cap.read();assert ok
            cap.release()
            p=HOST/'review_frames'/f"{row['task']}-{robot}-{row['condition']}-q{t:03d}.png";p.parent.mkdir(parents=True,exist_ok=True)
            assert cv2.imwrite(str(p),frame)
            with h5py.File(episode['telemetry_path'],'r') as f:
                starts=f['baseline/native_query_action_index'][:];q=int((starts==t).nonzero()[0][0]);native=f['baseline/native_query'][q].tolist()
            index.append(dict(task=row['task'],robot=robot,condition=int(row['condition']),query_action=t,absolute_path=str(p),bytes=p.stat().st_size,sha256=sha(p),source_video=str(video),source_video_sha256=sha(video),
                native_query=native,image_kind='PNG_ENCODED_FROM_ORIGINAL_LOSSY_MP4_FRAME_NOT_LOSSLESS_NETWORK_INPUT',intervention_applied=False,inference_calls=0))
    (OUT/'review_frames.json').write_text(json.dumps(dict(selection='All six descriptive history-caliper matches; source-initiation anchored, posthoc, not causal matches',images=index,altered_images=0,inference_calls=0),indent=2)+'\n')
    print(json.dumps([dict(path=r['absolute_path'],task=r['task'],robot=r['robot'],condition=r['condition'],query=r['query_action']) for r in index],indent=2))


if __name__=='__main__':main()
