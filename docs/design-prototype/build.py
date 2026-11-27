"""Build the offline preview from the editable design fragment. No dependencies."""
from pathlib import Path

root = Path(__file__).resolve().parent
fragment = (root / 'prototype.html').read_text()
head = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BetaBuddy · 锁扣伙伴设计稿</title><script src="assets/lucide.min.js"></script><style>body{margin:0;background:#f1f2ed;padding:28px;font-family:-apple-system,BlinkMacSystemFont,sans-serif}main{max-width:1040px;margin:auto}@media(max-width:600px){body{padding:10px}}</style></head><body><main>'''
(root / 'index.html').write_text(head + fragment + '</main></body></html>')
print('Updated docs/design-prototype/index.html')
