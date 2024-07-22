# # URL helpers, see https://github.com/NVlabs/stylegan
# # ------------------------------------------------------------------------------------------

# import requests
# import html
# import hashlib
# import gdown
# import glob
# import os
# import io
# from typing import Any
# import re
# import uuid

# weight_dic = {'afhqwild.pt': 'https://drive.google.com/file/d/14OnzO4QWaAytKXVqcfWo_o2MzoR4ygnr/view?usp=sharing',
#               'afhqdog.pt': 'https://drive.google.com/file/d/16v6jPtKVlvq8rg2Sdi3-R9qZEVDgvvEA/view?usp=sharing',
#               'afhqcat.pt': 'https://drive.google.com/file/d/1HXLER5R3EMI8DSYDBZafoqpX4EtyOf2R/view?usp=sharing',
#               'ffhq.pt': 'https://drive.google.com/file/d/1AT6bNR2ppK8f2ETL_evT27f3R_oyWNHS/view?usp=sharing',
#               'metfaces.pt': 'https://drive.google.com/file/d/16wM2PwVWzaMsRgPExvRGsq6BWw_muKbf/view?usp=sharing',
#               'seg.pth': 'https://drive.google.com/file/d/1lIKvQaFKHT5zC7uS4p17O9ZpfwmwlS62/view?usp=sharing'}


# def download_weight(weight_path):
#     gdown.download(weight_dic[os.path.basename(weight_path)],
#                    output=weight_path, fuzzy=True)


# def is_url(obj: Any) -> bool:
#     """Determine whether the given object is a valid URL string."""
#     if not isinstance(obj, str) or not "://" in obj:
#         return False
#     try:
#         res = requests.compat.urlparse(obj)
#         if not res.scheme or not res.netloc or not "." in res.netloc:
#             return False
#         res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
#         if not res.scheme or not res.netloc or not "." in res.netloc:
#             return False
#     except:
#         return False
#     return True


# def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True,
#              return_path: bool = False) -> Any:
#     """Download the given URL and return a binary-mode file object to access the data."""
#     assert is_url(url)
#     assert num_attempts >= 1

#     # Lookup from cache.
#     url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
#     if cache_dir is not None:
#         cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
#         if len(cache_files) == 1:
#             if (return_path):
#                 return cache_files[0]
#             else:
#                 return open(cache_files[0], "rb")

#     # Download.
#     url_name = None
#     url_data = None
#     with requests.Session() as session:
#         if verbose:
#             print("Downloading %s ..." % url, end="", flush=True)
#         for attempts_left in reversed(range(num_attempts)):
#             try:
#                 with session.get(url) as res:
#                     res.raise_for_status()
#                     if len(res.content) == 0:
#                         raise IOError("No data received")

#                     if len(res.content) < 8192:
#                         content_str = res.content.decode("utf-8")
#                         if "download_warning" in res.headers.get("Set-Cookie", ""):
#                             links = [html.unescape(link) for link in content_str.split('"') if
#                                      "export=download" in link]
#                             if len(links) == 1:
#                                 url = requests.compat.urljoin(url, links[0])
#                                 raise IOError("Google Drive virus checker nag")
#                         if "Google Drive - Quota exceeded" in content_str:
#                             raise IOError("Google Drive quota exceeded")

#                     match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
#                     url_name = match[1] if match else url
#                     url_data = res.content
#                     if verbose:
#                         print(" done")
#                     break
#             except:
#                 if not attempts_left:
#                     if verbose:
#                         print(" failed")
#                     raise
#                 if verbose:
#                     print(".", end="", flush=True)

#     # Save to cache.
#     if cache_dir is not None:
#         safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
#         cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
#         temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
#         os.makedirs(cache_dir, exist_ok=True)
#         with open(temp_file, "wb") as f:
#             f.write(url_data)
#         os.replace(temp_file, cache_file)  # atomic
#         if (return_path): return cache_file

#     # Return data as file object.
#     return io.BytesIO(url_data)


import gdown
import os
import requests
import html
import hashlib
import gdown
import glob
import io
from typing import Any
import re
import uuid

weight_dic = {
    'afhqwild.pt': 'https://drive.google.com/uc?id=14OnzO4QWaAytKXVqcfWo_o2MzoR4ygnr',
    'afhqdog.pt': 'https://drive.google.com/uc?id=16v6jPtKVlvq8rg2Sdi3-R9qZEVDgvvEA',
    'afhqcat.pt': 'https://drive.google.com/uc?id=1HXLER5R3EMI8DSYDBZafoqpX4EtyOf2R',
    'ffhq.pt': 'https://drive.google.com/uc?id=1AT6bNR2ppK8f2ETL_evT27f3R_oyWNHS',
    'metfaces.pt': 'https://drive.google.com/uc?id=16wM2PwVWzaMsRgPExvRGsq6BWw_muKbf',
    'seg.pth': 'https://drive.google.com/uc?id=1lIKvQaFKHT5zC7uS4p17O9ZpfwmwlS62'
}

def download_weight(weight_path):
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    if not os.path.exists(weight_path):
        gdown.download(weight_dic[os.path.basename(weight_path)], weight_path, quiet=False)

def is_url(obj: Any) -> bool:
    """Determine whether the given object is a valid URL string."""
    if not isinstance(obj, str) or not "://" in obj:
        return False
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True

def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_path: bool = False) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert is_url(url)
    assert num_attempts >= 1

    # Lookup from cache.
    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache_dir is not None:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            if return_path:
                return cache_files[0]
            else:
                return open(cache_files[0], "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive quota exceeded")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache_dir is not None:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file)  # atomic
        if return_path:
            return cache_file

    # Return data as file object.
    return io.BytesIO(url_data)
