"""Local-only design preview. Reads GymPack in place; never copies or modifies it."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit, unquote
import json, mimetypes, os, argparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACK = Path(os.environ.get('BETABUDDY_GYM_PACK', ROOT / 'ios' / 'GymPack')).resolve()
try:
    manifest = json.loads((PACK / 'manifest.json').read_text())
    files = manifest.get('files', [])
    ALLOWED = set(files if isinstance(files, dict) else [f['path'] for f in files])
    ENTRY = manifest.get('entry', 'index.html')
    ALLOWED.add(ENTRY)
except (OSError, ValueError, KeyError):
    manifest, ALLOWED, ENTRY = {}, set(), 'index.html'

class Preview(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Keep private scene names and request paths out of logs.

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path == '/config.js':
            data = {'pack':bool(ALLOWED), 'analysis':'analysis.html' in ALLOWED,
                    'capture':'capture.html' in ALLOWED,
                    'memberVenues':[v['id'] for v in manifest.get('venues', [])
                        if v['id'] != (manifest.get('defaultVenue') or next((x['id'] for x in manifest.get('venues', []) if x.get('default')), None) or next(iter(manifest.get('venues', [])), {}).get('id'))]}
            return self.reply(('window.BB_PREVIEW='+json.dumps(data)+';').encode(), 'text/javascript')
        if path.startswith('/gym/'):
            rel = path[5:] or ENTRY
            if rel == 'index.html': rel = ENTRY
            target = (PACK / rel).resolve()
            if rel not in ALLOWED or not target.is_relative_to(PACK) or not target.is_file():
                return self.send_error(404)
            payload = target.read_bytes()
            if rel == ENTRY:
                html = payload.decode('utf-8')
                inject = '<link rel="stylesheet" href="/gym-theme.css"><script src="/gym-bridge.js"></script>'
                html = html.replace('</head>', inject+'</head>', 1)
                payload = html.encode()
            elif rel == 'analysis.html':
                html = payload.decode('utf-8').replace('</head>', '<link rel="stylesheet" href="/analysis-theme.css"></head>', 1)
                payload = html.encode()
            return self.reply(payload, mimetypes.guess_type(str(target))[0] or 'application/octet-stream')
        if path == '/icons.js':
            return self.reply((HERE.parent / 'assets' / 'lucide.min.js').read_bytes(), 'text/javascript')
        if path in ['/', '/index.html']:
            return self.reply((HERE / 'index.html').read_bytes(), 'text/html')
        rel = path.lstrip('/')
        target = (HERE / rel).resolve()
        # Source and private filesystem paths are not exposed by the preview server.
        if target.is_relative_to(HERE) and target.is_file() and target.suffix in {'.css','.js','.svg','.png','.woff2'} and not any(p.startswith('.') for p in Path(rel).parts):
            return self.reply(target.read_bytes(), mimetypes.guess_type(str(target))[0] or 'application/octet-stream')
        self.send_error(404)

    def reply(self, payload, mime):
        self.send_response(200)
        self.send_header('Content-Type', mime+'; charset=utf-8' if mime.startswith('text/') else mime)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self' data: blob:; script-src 'self' 'unsafe-inline' 'unsafe-eval' blob:; style-src 'self' 'unsafe-inline'; connect-src 'self' blob: data:; worker-src 'self' blob:; frame-ancestors 'self'; object-src 'none'")
        self.end_headers()
        try: self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError): pass

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8767)
    args=parser.parse_args()
    print(f'BetaBuddy design preview: http://127.0.0.1:{args.port}',flush=True)
    try:
        ThreadingHTTPServer(('127.0.0.1',args.port),Preview).serve_forever()
    except KeyboardInterrupt:
        pass
