import urllib.request
import re

def get_headers(table):
    req = urllib.request.Request(f'https://www.tennisabstract.com/cgi-bin/wplayer-more.cgi?p=200033/ArynaSabalenka&table={table}', headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read().decode('utf-8')
    table_match = re.search(r'<table[^>]*>(.*?)</table>', html, re.DOTALL)
    if not table_match:
        return f"No table for {table}\n"
    row_match = re.search(r'<tr[^>]*>(.*?)</tr>', table_match.group(1), re.DOTALL)
    if not row_match:
        return f"No row for {table}\n"
    cells = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', row_match.group(1), re.DOTALL)
    headers = [re.sub(r'<[^>]+>', '', c).strip().replace('\xa0', ' ') for c in cells]
    return f"{table} headers: {headers}\n"

with open('headers.txt', 'w') as f:
    for t in ['mcp-serve', 'mcp-return', 'mcp-rally', 'mcp-tactics', 'winners-errors']:
        try:
            f.write(get_headers(t))
        except Exception as e:
            f.write(f"Failed {t}: {e}\n")
