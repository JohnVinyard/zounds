import requests
import urlparse
import os


def ensure_local_file(remote_url, local_path, chunksize=4096):
    parsed = urlparse.urlparse(remote_url)
    filename = os.path.split(parsed.path)[-1]
    local = os.path.join(local_path, filename)

    if not os.path.exists(local):
        with open(local, 'wb') as f:
            resp = requests.get(remote_url, stream=True)
            total_bytes = int(resp.headers['Content-Length'])
            for i, chunk in enumerate(resp.iter_content(chunk_size=chunksize)):
                f.write(chunk)
                progress = ((i * chunksize) / float(total_bytes)) * 100
                print '{remote_url} {progress:.2f}% complete'.format(**locals())

    return local
