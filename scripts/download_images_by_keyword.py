"""Download images for a given keyword using an image-search API.

This script does NOT call any API unless you provide a valid API key and endpoint.
It supports Microsoft/Bing Image Search (Azure) and Google Custom Search (optional).

Usage examples (Bing):
  python scripts/download_images_by_keyword.py --query "manufacturing worker" --limit 50 \
      --engine bing --api-key YOUR_KEY --endpoint https://api.bing.microsoft.com/v7.0/images/search --out data/images/manufacturing

If you don't have an API key, you can manually put target images into `data/images/manufacturing/` and the pipeline will process them.
"""
import argparse
from pathlib import Path
import os
import sys
import time
import hashlib
import mimetypes

try:
    import requests
except Exception:
    requests = None


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def download_url(url, outpath, session=None, timeout=10):
    s = session or requests
    try:
        resp = s.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get('content-type', '')
        ext = mimetypes.guess_extension(content_type.split(';')[0].strip()) or '.jpg'
        outfile = outpath.with_suffix(ext)
        with open(outfile, 'wb') as fh:
            for chunk in resp.iter_content(1024 * 8):
                fh.write(chunk)
        return outfile
    except Exception as e:
        return None


def bing_search(query, api_key, endpoint, limit=50):
    headers = { 'Ocp-Apim-Subscription-Key': api_key }
    params = { 'q': query, 'count': min(limit, 50), 'safeSearch': 'Moderate' }
    r = requests.get(endpoint, headers=headers, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    imgs = [i.get('contentUrl') for i in data.get('value', []) if i.get('contentUrl')]
    return imgs[:limit]


def google_search(query, api_key, cx, limit=50):
    # Requires `cx` (Custom Search Engine id) and `api_key`.
    urls = []
    base = 'https://www.googleapis.com/customsearch/v1'
    params = {'q': query, 'searchType': 'image', 'key': api_key, 'cx': cx, 'num': 10}
    start = 1
    while len(urls) < limit:
        params['start'] = start
        r = requests.get(base, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        items = data.get('items', [])
        for it in items:
            link = it.get('link')
            if link:
                urls.append(link)
                if len(urls) >= limit:
                    break
        if not items:
            break
        start += len(items)
    return urls[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', required=True)
    parser.add_argument('--engine', choices=['bing','google'], default='bing')
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--endpoint', default=None, help='Bing endpoint e.g. https://api.bing.microsoft.com/v7.0/images/search')
    parser.add_argument('--cx', default=None, help='Google Custom Search CX id (if using google)')
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--out', default='data/images/manufacturing')
    args = parser.parse_args()

    outdir = Path(args.out)
    safe_mkdir(outdir)

    if requests is None:
        print('requests not installed in this environment. Please install requests or download images manually.')
        sys.exit(1)

    urls = []
    if args.engine == 'bing':
        if not args.api_key or not args.endpoint:
            print('Bing engine requires --api-key and --endpoint. See script help for usage.')
            sys.exit(1)
        print('Searching Bing for:', args.query)
        urls = bing_search(args.query, args.api_key, args.endpoint, limit=args.limit)
    else:
        if not args.api_key or not args.cx:
            print('Google engine requires --api-key and --cx (Custom Search Engine id).')
            sys.exit(1)
        print('Searching Google for:', args.query)
        urls = google_search(args.query, args.api_key, args.cx, limit=args.limit)

    print(f'Found {len(urls)} image URLs. Starting download...')
    session = requests
    success = 0
    for i, url in enumerate(urls, start=1):
        try:
            h = hashlib.sha1(url.encode('utf-8')).hexdigest()[:10]
            outpath = outdir / f'{i:04d}_{h}'
            res = download_url(url, outpath, session=session)
            if res:
                print(f'[{i}/{len(urls)}] Saved: {res.name}')
                success += 1
            else:
                print(f'[{i}/{len(urls)}] Failed: {url}')
        except Exception as e:
            print('Error downloading url:', e)
        time.sleep(0.1)

    print(f'Downloaded {success}/{len(urls)} images to {outdir}')


if __name__ == '__main__':
    main()
