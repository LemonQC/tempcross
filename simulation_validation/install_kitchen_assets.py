"""Resume official, SHA256-verified kitchen archives without changing assets.

Run inside the BARX environment. Extraction uses the upstream safety checks and
installation code; only the HTTP transfer is replaced by resumable byte ranges.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import tempfile
import urllib.request

from simulation_validation.download_ranges import download


def install(archive_root, workers=4):
    from robocasa.scripts.download_kitchen_assets import (
        DOWNLOAD_ASSET_REGISTRY, _safe_extract, _install_extracted_tree,
    )
    root=Path(archive_root).resolve();root.mkdir(parents=True,exist_ok=True)
    def one(pair):
        name,item=pair;archive=root/(name+'.zip')
        # Resolve Box redirects once instead of once per 8 MiB range.
        # The final archive remains checked against the upstream SHA256.
        request=urllib.request.Request(item['url'],headers={'Range':'bytes=0-0'})
        with urllib.request.urlopen(request,timeout=120) as response:
            if response.status!=206 or response.headers.get('Content-Range')!=f"bytes 0-0/{item['size_bytes']}":
                raise RuntimeError('Unexpected official archive range metadata')
            transport=response.geturl()
            response.read(1)
        download(transport,archive,item['size_bytes'],item['sha256'],workers)
        folder=Path(item['folder']);folder.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.coft-verified-',dir=folder.parent) as temporary:
            _safe_extract(archive,temporary)
            _install_extracted_tree(temporary,folder)
        record=dict(name=name,archive=str(archive),installed_path=str(folder),
                    sha256=item['sha256'],size_bytes=item['size_bytes'],status='INSTALLED_VERIFIED_ARCHIVE')
        (root/(name+'.installed.json')).write_text(json.dumps(record,indent=2))
        print(json.dumps(record),flush=True)
        return record
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(one,DOWNLOAD_ASSET_REGISTRY.items()))
    (root/'installation.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive-root',required=True)
    parser.add_argument('--workers-per-archive',type=int,default=4);args=parser.parse_args()
    install(args.archive_root,args.workers_per_archive)
