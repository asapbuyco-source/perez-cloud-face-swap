"""Downloads the face-swap models (too big for git). Run at build time."""
import os
import urllib.request

URLS = {
    "models/inswapper_128.onnx": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    "models/w600k_r50.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
}

os.makedirs("models", exist_ok=True)
for path, url in URLS.items():
    if os.path.exists(path) and os.path.getsize(path) > 100_000_000:
        print(f"skip {path} (already present)")
        continue
    print(f"downloading {url.split('/')[2] if 'github' in url else 'huggingface'} -> {path}")
    if "buffalo_l.zip" in url:
        import io
        import zipfile
        data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})).read()
        z = zipfile.ZipFile(io.BytesIO(data))
        for name in z.namelist():
            if "w600k_r50.onnx" in name:
                open(path, "wb").write(z.read(name))
                break
    else:
        urllib.request.urlretrieve(url, path)
print("done")