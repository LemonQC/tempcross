"""Index actual upstream videos; optionally generate external contact sheets.
An empty index is explicit when P0 produced no rollout. Never fabricates videos.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--sheets',action='store_true');p.add_argument('--reviews');a=p.parse_args()
    root=Path(a.root).resolve();rows=[]
    reviews={r['video_path']:r for r in json.loads(Path(a.reviews).read_text())} if a.reviews else {}
    for video in sorted(root.rglob('*.mp4')) if root.exists() else []:
        row=dict(path=str(video),bytes=video.stat().st_size,review_status='SAMPLED_FRAMES_AND_STATE_REVIEWED' if str(video) in reviews else 'NOT_REVIEWED')
        if a.sheets:
            executable=shutil.which('ffmpeg')
            if not executable:
                try:
                    import imageio_ffmpeg
                    executable=imageio_ffmpeg.get_ffmpeg_exe()
                except ImportError as error:
                    raise RuntimeError('ffmpeg or imageio_ffmpeg required for contact sheets') from error
            sheet=video.with_suffix('.contact-sheet.jpg')
            subprocess.run([executable,'-nostdin','-y','-i',str(video),'-vf','fps=1/2,scale=320:-1,tile=4x4','-frames:v','1',str(sheet)],check=True,capture_output=True)
            row['contact_sheet']=str(sheet)
        rows.append(row)
    Path(a.out).write_text(json.dumps(dict(root=str(root),count=len(rows),videos=rows),indent=2))
if __name__=='__main__':main()
