import requests
from tqdm import tqdm
print('getting val.bin')
url = "https://huggingface.co/datasets/Vineet314/LLM-fineweb/resolve/main/val.bin"
output_file = "val.bin"
response = requests.head(url, allow_redirects=True)
total_size = int(response.headers.get('content-length', 0))

# Stream download with progress bar
with requests.get(url, stream=True) as r, open(output_file, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=output_file,dynamic_ncols=True) as bar:

    for chunk in r.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
            bar.update(len(chunk))

print('getting train.bin')
url = "https://huggingface.co/datasets/Vineet314/LLM-fineweb/resolve/main/train.bin"
output_file = "train.bin"
response = requests.head(url, allow_redirects=True)
total_size = int(response.headers.get('content-length', 0))

with requests.get(url, stream=True) as r, open(output_file, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=output_file,dynamic_ncols=True) as bar:

    for chunk in r.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
            bar.update(len(chunk))
