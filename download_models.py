"""Downloads the face-swap models (too big for git). Run at build time."""
import io
import os
import urllib.request
import zipfile

URLS = {
    "models/inswapper_128.onnx": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    "models/inswapper_128_fp16.onnx": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx",
    "models/w600k_r50.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "models/gfpgan_1.4.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx",
}

os.makedirs("models", exist_ok=True)

def already(path):
    return os.path.exists(path) and os.path.getsize(path) > 10_000_000


def download(url, dest):
    print(f"downloading -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    if "buffalo_l.zip" in url:
        z = zipfile.ZipFile(io.BytesIO(data))
        for name in z.namelist():
            if name.endswith(".onnx"):
                out = os.path.join("models", "buffalo_l", os.path.basename(name))
                os.makedirs(os.path.dirname(out), exist_ok=True)
                open(out, "wb").write(z.read(name))
                print(f"  extracted {out}")
    else:
        with open(dest, "wb") as f:
            f.write(data)
    print(f"  done ({len(data) / 1e6:.1f} MB)")


for path, url in URLS.items():
    if already(path):
        print(f"skip {path} (already present)")
        continue
    try:
        download(url, path)
    except Exception as e:  # noqa: BLE001
        print(f"WARN {path} failed: {e}")

# buffalo_l directory check (insightface expects models/buffalo_l/*.onnx)
buff_dir = "models/buffalo_l"
if not (os.path.isdir(buff_dir) and any(f.endswith(".onnx") for f in os.listdir(buff_dir))):
    print("buffalo_l missing - downloading zip")
    try:
        download(URLS["models/w600k_r50.onnx"], "models/buffalo_l.zip")
    except Exception as e:  # noqa: BLE001
        print(f"WARN buffalo_l failed: {e}")

print("done")