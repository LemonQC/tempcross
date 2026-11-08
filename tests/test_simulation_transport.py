import hashlib
import io
import re
from pathlib import Path
import pytest
from simulation_validation import download_ranges as transport

class Response(io.BytesIO):
    status=206
    def __init__(self,data,lo,hi,total):
        super().__init__(data);self.headers={'Content-Range':f'bytes {lo}-{hi}/{total}'}


def test_interrupted_range_resumes_and_verifies_full_artifact(tmp_path,monkeypatch):
    data=b'BARX'*((8*1024*1024+1024)//4);destination=tmp_path/'artifact.bin';calls=[];interrupted=False
    def open_request(request,timeout):
        nonlocal interrupted
        lo,hi=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',request.get_header('Range')).groups())
        calls.append((lo,hi));body=data[lo:hi+1]
        if lo==0 and not interrupted:
            body=body[:4096];interrupted=True
        return Response(body,lo,hi,len(data))
    monkeypatch.setattr(transport.urllib.request,'urlopen',open_request)
    monkeypatch.setattr(transport.time,'sleep',lambda _:None)
    transport.download('https://official.example/file',destination,len(data),hashlib.sha256(data).hexdigest(),2)
    assert destination.read_bytes()==data
    assert any(lo==4096 for lo,hi in calls)


def test_invalid_range_response_is_rejected(tmp_path,monkeypatch):
    def incorrect(request,timeout):return Response(b'bad',1,3,4)
    monkeypatch.setattr(transport.urllib.request,'urlopen',incorrect)
    monkeypatch.setattr(transport.time,'sleep',lambda _:None)
    with pytest.raises(RuntimeError,match='Invalid range'):
        transport.download('https://official.example/file',tmp_path/'artifact.bin',4,'0'*64,1)
    assert not (tmp_path/'artifact.bin').exists()


def test_wrong_full_sha_never_installs_an_artifact(tmp_path,monkeypatch):
    monkeypatch.setattr(transport.urllib.request,'urlopen',lambda request,timeout:Response(b'COFT',0,3,4))
    with pytest.raises(AssertionError,match='SHA256'):
        transport.download('https://official.example/file',tmp_path/'artifact.bin',4,'0'*64,1)
    assert not (tmp_path/'artifact.bin').exists()


def test_presigned_url_query_is_never_modified(tmp_path,monkeypatch):
    url='https://official.example/file?X-Amz-Signature=recorded-signature'
    def response(request,timeout):
        assert request.full_url==url
        return Response(b'COFT',0,3,4)
    monkeypatch.setattr(transport.urllib.request,'urlopen',response)
    transport.download(url,tmp_path/'artifact.bin',4,hashlib.sha256(b'COFT').hexdigest(),1)
