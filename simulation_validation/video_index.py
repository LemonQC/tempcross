"""Index actual upstream videos; optionally generate external contact sheets.
An empty index is explicit when P0 produced no rollout. Never fabricates videos.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--sheets',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve();rows=[]
    for video in sorted(root.rglob('*.mp4')) if root.exists() else []:
        row=dict(path=str(video),bytes=video.stat().st_size,review_status='NOT_REVIEWED')
        if a.sheets:
            executable=shutil.which('ffmpeg')
            if not executable:raise RuntimeError('ffmpeg not available for contact sheets')
            sheet=video.with_suffix('.contact-sheet.jpg')
            subprocess.run([executable,'-nostdin','-y','-i',str(video),'-vf','fps=1/2,scale=320:-1,tile=4x4','-frames:v','1',str(sheet)],check=True,capture_output=True)
            row['contact_sheet']=str(sheet)
        rows.append(row)
    Path(a.out).write_text(json.dumps(dict(root=str(root),count=len(rows),videos=rows),indent=2))
if __name__=='__main__':main()
