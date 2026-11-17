#!/usr/bin/env python3
"""
GitHub Actions Bulut İndirici (cloud_runner.py)
Bilgisayar kapalıyken GitHub sunucularında çalışır:
1) CHELSA-daily v2.1 (1979-2025) COG karolarını HTTP-Range ile çeker ve Türkiye Ege/Akdeniz (636x1380) alanına kırpar.
2) Her yılı sıkıştırılmış .npz kübü (CHELSA_{var}_{year}_EgeAkdeniz_1km.npz) olarak kaydeder.
3) Otomatik olarak hem GitHub Release ('chelsa-v2.1-data') altına yükler (hiçbir dış şifre/token gerektirmez),
   hem de eğer GDRIVE_RCLONE_TOKEN tanımlıysa doğrudan Google Drive'a aktarır.
"""
import os
import re
import sys
import time
import zlib
import struct
import threading
import subprocess
import http.client
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import numpy as np

COL_START, COL_END = 24660, 26040  # 1380 sütun (25.50°D - 37.00°D)
ROW_START, ROW_END = 5183, 5819    # 636 satır  (35.51°K - 40.81°K)
REL_C0, REL_C1 = COL_START - 48 * 512, COL_END - 48 * 512
REL_R0, REL_R1 = ROW_START - 10 * 512, ROW_END - 10 * 512

LON0, LAT0, DX, DY = -180.00013888885002, 83.99986041515001, 0.0083333333, 0.0083333333
LONS_1KM = LON0 + (np.arange(COL_START, COL_END) + 0.5) * DX
LATS_1KM = LAT0 - (np.arange(ROW_START, ROW_END) + 0.5) * DY
CLT_REL_COLS = ((LONS_1KM - (-180.02485599000002)) / 0.02499999).astype(np.int32) - 16 * 512
CLT_REL_ROWS = ((89.99992800000001 - LATS_1KM) / 0.02499999).astype(np.int32) - 3 * 512

tls = threading.local()


def get_conn():
    if not hasattr(tls, "conn") or tls.conn is None:
        tls.conn = http.client.HTTPSConnection("os.unil.cloud.switch.ch", timeout=25)
    return tls.conn


def reset_conn():
    if hasattr(tls, "conn") and tls.conn is not None:
        try:
            tls.conn.close()
        except Exception:
            pass
    tls.conn = http.client.HTTPSConnection("os.unil.cloud.switch.ch", timeout=25)
    return tls.conn


def list_year_keys(var, year):
    url = f"https://os.unil.cloud.switch.ch/chelsa02/?prefix=chelsa/global/daily/{var}/{year}/&max-keys=1000"
    req = urllib.request.Request(url, headers={"User-Agent": "TUBITAK-1002-GitHubActions/1.0"})
    xml_bytes = urllib.request.urlopen(req, timeout=20).read()
    root = ET.fromstring(xml_bytes.decode("utf-8"))
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    items = []
    for k in root.findall("s3:Contents", ns):
        key = k.find("s3:Key", ns).text
        size = int(k.find("s3:Size", ns).text)
        if key.endswith(".tif") and size > 0:
            m = re.search(r"_(\d{2})_(\d{2})_(\d{4})_V\.2\.1\.tif$", key)
            if m:
                dd, mm, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))
                items.append((f"{yyyy:04d}-{mm:02d}-{dd:02d}", key))
    items.sort(key=lambda x: x[0])
    return items


def fetch_and_crop_day(item):
    dt_str, key = item
    path = "/chelsa02/" + key
    for attempt in range(4):
        try:
            conn = get_conn() if attempt == 0 else reset_conn()
            conn.request("GET", path, headers={"Range": "bytes=0-32767", "Connection": "keep-alive"})
            r1 = conn.getresponse()
            buf = r1.read()
            if r1.status not in (200, 206) or len(buf) < 10000:
                raise RuntimeError(f"Header status {r1.status}")

            ifd_offset = struct.unpack("<I", buf[4:8])[0]
            num_entries = struct.unpack("<H", buf[ifd_offset:ifd_offset + 2])[0]
            off_tag, cnt_tag, ntiles = 3376, 17316, 3485
            for i in range(num_entries):
                pos = ifd_offset + 2 + i * 12
                tag, _, count, val_off = struct.unpack("<HHII", buf[pos:pos + 12])
                if tag == 324:
                    off_tag = val_off
                    ntiles = count
                elif tag == 325:
                    cnt_tag = val_off

            tile_offsets = struct.unpack(f"<{ntiles}I", buf[off_tag:off_tag + ntiles * 4])
            tile_counts = struct.unpack(f"<{ntiles}I", buf[cnt_tag:cnt_tag + ntiles * 4])

            if ntiles == 435:
                s_top, e_top = tile_offsets[103], tile_offsets[103] + tile_counts[103] - 1
                s_bot, e_bot = tile_offsets[132], tile_offsets[132] + tile_counts[132] - 1
                conn.request("GET", path, headers={"Range": f"bytes={s_top}-{e_top}", "Connection": "keep-alive"})
                b_top = conn.getresponse().read()
                conn.request("GET", path, headers={"Range": f"bytes={s_bot}-{e_bot}", "Connection": "keep-alive"})
                b_bot = conn.getresponse().read()
                canvas_clt = np.empty((1024, 512), dtype=np.uint16)
                canvas_clt[0:512, :] = np.cumsum(np.frombuffer(zlib.decompress(b_top), dtype="<u2").reshape(512, 512), axis=1, dtype=np.uint16)
                canvas_clt[512:1024, :] = np.cumsum(np.frombuffer(zlib.decompress(b_bot), dtype="<u2").reshape(512, 512), axis=1, dtype=np.uint16)
                return (dt_str, canvas_clt[np.ix_(CLT_REL_ROWS, CLT_REL_COLS)].copy(), len(buf) + len(b_top) + len(b_bot))

            s10, e10 = tile_offsets[898], tile_offsets[900] + tile_counts[900] - 1
            s11, e11 = tile_offsets[983], tile_offsets[985] + tile_counts[985] - 1

            conn.request("GET", path, headers={"Range": f"bytes={s10}-{e10}", "Connection": "keep-alive"})
            b10 = conn.getresponse().read()
            conn.request("GET", path, headers={"Range": f"bytes={s11}-{e11}", "Connection": "keep-alive"})
            b11 = conn.getresponse().read()

            canvas = np.empty((1024, 1536), dtype=np.uint16)
            for tx_i, t_idx in enumerate([898, 899, 900]):
                rel_s = tile_offsets[t_idx] - s10
                raw = zlib.decompress(b10[rel_s:rel_s + tile_counts[t_idx]])
                canvas[0:512, tx_i * 512:(tx_i + 1) * 512] = np.cumsum(np.frombuffer(raw, dtype="<u2").reshape(512, 512), axis=1, dtype=np.uint16)

            for tx_i, t_idx in enumerate([983, 984, 985]):
                rel_s = tile_offsets[t_idx] - s11
                raw = zlib.decompress(b11[rel_s:rel_s + tile_counts[t_idx]])
                canvas[512:1024, tx_i * 512:(tx_i + 1) * 512] = np.cumsum(np.frombuffer(raw, dtype="<u2").reshape(512, 512), axis=1, dtype=np.uint16)

            return (dt_str, canvas[REL_R0:REL_R1, REL_C0:REL_C1].copy(), len(buf) + len(b10) + len(b11))
        except Exception:
            if attempt == 3:
                raise
            time.sleep(0.5 * (attempt + 1))


def main():
    var = sys.argv[1]
    y_start = int(sys.argv[2]) if len(sys.argv) > 2 else 1979
    y_end = int(sys.argv[3]) if len(sys.argv) > 3 else 2025
    release_tag = "chelsa-v2.1-data"

    outside_mask = None
    if os.path.exists("calisma_alani_statik_grid_1km.npz"):
        st = np.load("calisma_alani_statik_grid_1km.npz")
        outside_mask = (st["province_id"] == 0)

    # Check already uploaded assets in GitHub Release
    existing_assets = set()
    res_check = subprocess.run(["gh", "release", "view", release_tag, "--json", "assets"], capture_output=True, text=True)
    if res_check.returncode == 0:
        import json
        for a in json.loads(res_check.stdout).get("assets", []):
            existing_assets.add(a["name"])
    else:
        subprocess.run(["gh", "release", "create", release_tag, "--title", "CHELSA-daily v2.1 Ege & Akdeniz 1km Küpleri (1979-2025)", "--notes", "Otomatik bulut indirme çıktıları."], check=False)

    for year in range(y_start, y_end + 1):
        if var == "pr" and year >= 2020:
            continue
        if var == "clt" and year == 2007:
            continue
        out_fname = f"CHELSA_{var}_{year}_EgeAkdeniz_1km.npz"
        if out_fname in existing_assets:
            print(f"[ATLANDI] {out_fname} zaten GitHub Release içinde mevcut.", flush=True)
            continue

        items = list_year_keys(var, year)
        if not items:
            continue

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=32) as ex:
            results = list(ex.map(fetch_and_crop_day, items))

        dates = np.array([r[0] for r in results], dtype="U10")
        cube = np.stack([r[1] for r in results], axis=0)
        if outside_mask is not None:
            cube[:, outside_mask] = 65535

        np.savez_compressed(
            out_fname,
            dates=dates,
            data=cube,
            variable=np.array(var),
            year=np.array(year, dtype=np.int32),
            col_bounds=np.array([COL_START, COL_END], dtype=np.int32),
            row_bounds=np.array([ROW_START, ROW_END], dtype=np.int32),
        )
        mb = os.path.getsize(out_fname) / 1e6
        print(f"[OK] {out_fname} ({mb:.1f} MB) {time.time() - t0:.1f} saniyede oluşturuldu.", flush=True)

        # 1. Upload to GitHub Release (Zero external secrets needed!)
        subprocess.run(["gh", "release", "upload", release_tag, out_fname, "--clobber"], check=False)

        # 2. Optional: If GDRIVE_RCLONE_TOKEN is configured, also copy directly to Google Drive
        if os.environ.get("RCLONE_CONFIG_GDRIVE_TOKEN"):
            subprocess.run(["rclone", "copy", out_fname, f"gdrive:TUBITAK_1002_CHELSA_HCI/02_CHELSA_Gunluk_Veri_1979_2025/{var}/"], check=False)

        os.remove(out_fname)


if __name__ == "__main__":
    main()
