#!/usr/bin/env python3
"""
GitHub Actions Bulut İndirici (cloud_runner.py) - v2 (Classic TIFF + BigTIFF + Deflate + LZW Destekli)
Bilgisayar kapalıyken GitHub sunucularında çalışır:
1) CHELSA-daily v2.1 (1979-2025) COG karolarını HTTP-Range ile çeker (hem Classic TIFF 'II*' hem BigTIFF 'II+',
   hem Deflate [comp=8] hem LZW [comp=5] sıkıştırmalarını ve dosya sonu IFD'lerini destekler).
2) Her yılı sıkıştırılmış .npz kübü (CHELSA_{var}_{year}_EgeAkdeniz_1km.npz) olarak kaydeder.
3) Otomatik olarak GitHub Release ('chelsa-v2.1-data') altına yükler (mevcut yılları atlar).
"""
import os
import re
import sys
import json
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

try:
    import imagecodecs
    HAS_IMAGECODECS = True
except ImportError:
    HAS_IMAGECODECS = False

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
        tls.conn = http.client.HTTPSConnection("os.unil.cloud.switch.ch", timeout=30)
    return tls.conn


def reset_conn():
    if hasattr(tls, "conn") and tls.conn is not None:
        try:
            tls.conn.close()
        except Exception:
            pass
    tls.conn = http.client.HTTPSConnection("os.unil.cloud.switch.ch", timeout=30)
    return tls.conn


def fetch_range(conn, path, start_b, end_b):
    expected = end_b - start_b + 1
    conn.request("GET", path, headers={"Range": f"bytes={start_b}-{end_b}", "Connection": "keep-alive"})
    resp = conn.getresponse()
    data = resp.read()
    if resp.status not in (200, 206) or len(data) < expected:
        raise RuntimeError(f"HTTP {resp.status}: got {len(data)}/{expected} bytes")
    return data, resp.getheader("Content-Range", "")


def decompress_tile(comp_bytes, comp_type, predictor):
    if comp_type == 8 or comp_type == 32946:
        raw = zlib.decompress(comp_bytes)
    elif comp_type == 5:
        if HAS_IMAGECODECS:
            raw = imagecodecs.lzw_decode(comp_bytes)
        else:
            raise RuntimeError("LZW tile requires imagecodecs")
    elif comp_type == 1:
        raw = comp_bytes
    else:
        raise RuntimeError(f"Unsupported TIFF compression: {comp_type}")

    arr = np.frombuffer(raw, dtype="<u2").reshape(512, 512)
    if predictor == 2:
        arr = np.cumsum(arr, axis=1, dtype=np.uint16)
    return arr


def parse_ifd_and_tiles(conn, path):
    buf0, crange = fetch_range(conn, path, 0, 32767)
    total_len = int(crange.split("/")[-1]) if "/" in crange else None
    magic = buf0[:4]

    if magic == b"II*\x00":
        # Classic 32-bit TIFF
        ifd_off = struct.unpack("<I", buf0[4:8])[0]
        if ifd_off + 20000 <= len(buf0):
            ifd_buf = buf0
            base_off = 0
        else:
            end_b = min(total_len - 1, ifd_off + 65535) if total_len else ifd_off + 65535
            ifd_buf, _ = fetch_range(conn, path, ifd_off, end_b)
            base_off = ifd_off

        rel_ifd = ifd_off - base_off
        num_entries = struct.unpack("<H", ifd_buf[rel_ifd:rel_ifd + 2])[0]
        comp_type, predictor = 8, 2
        off_tag, cnt_tag, ntiles, off_type, cnt_type = 0, 0, 3485, 4, 4
        for i in range(num_entries):
            pos = rel_ifd + 2 + i * 12
            tag, ttype, count, val_off = struct.unpack("<HHII", ifd_buf[pos:pos + 12])
            if tag == 259:
                comp_type = val_off
            elif tag == 317:
                predictor = val_off
            elif tag in (324, 273):
                off_tag, ntiles, off_type = val_off, count, ttype
            elif tag in (325, 279):
                cnt_tag, cnt_type = val_off, ttype

        off_elem_sz = 8 if off_type == 16 else 4
        cnt_elem_sz = 8 if cnt_type == 16 else 4
        off_fmt = f"<{ntiles}{'Q' if off_elem_sz == 8 else 'I'}"
        cnt_fmt = f"<{ntiles}{'Q' if cnt_elem_sz == 8 else 'I'}"

        if off_tag >= base_off and off_tag + ntiles * off_elem_sz <= base_off + len(ifd_buf):
            b_off = ifd_buf[off_tag - base_off:off_tag - base_off + ntiles * off_elem_sz]
        else:
            b_off, _ = fetch_range(conn, path, off_tag, off_tag + ntiles * off_elem_sz - 1)

        if cnt_tag >= base_off and cnt_tag + ntiles * cnt_elem_sz <= base_off + len(ifd_buf):
            b_cnt = ifd_buf[cnt_tag - base_off:cnt_tag - base_off + ntiles * cnt_elem_sz]
        else:
            b_cnt, _ = fetch_range(conn, path, cnt_tag, cnt_tag + ntiles * cnt_elem_sz - 1)

        tile_offsets = struct.unpack(off_fmt, b_off)
        tile_counts = struct.unpack(cnt_fmt, b_cnt)
        return ntiles, comp_type, predictor, tile_offsets, tile_counts

    elif magic == b"II+\x00":
        # 64-bit BigTIFF
        _, _, ifd_off = struct.unpack("<HHQ", buf0[4:16])
        if ifd_off + 36000 <= len(buf0):
            ifd_buf = buf0
            base_off = 0
        else:
            end_b = min(total_len - 1, ifd_off + 65535) if total_len else ifd_off + 65535
            ifd_buf, _ = fetch_range(conn, path, ifd_off, end_b)
            base_off = ifd_off

        rel_ifd = ifd_off - base_off
        num_entries = struct.unpack("<Q", ifd_buf[rel_ifd:rel_ifd + 8])[0]
        comp_type, predictor = 8, 2
        off_tag, cnt_tag, ntiles, off_type, cnt_type = 0, 0, 3485, 16, 4
        for i in range(num_entries):
            pos = rel_ifd + 8 + i * 20
            tag, ttype, count, val_off = struct.unpack("<HHQQ", ifd_buf[pos:pos + 20])
            if tag == 259:
                comp_type = val_off
            elif tag == 317:
                predictor = val_off
            elif tag in (324, 273):
                off_tag, ntiles, off_type = val_off, count, ttype
            elif tag in (325, 279):
                cnt_tag, cnt_type = val_off, ttype

        off_elem_sz = 8 if off_type == 16 else 4
        cnt_elem_sz = 8 if cnt_type == 16 else 4
        off_fmt = f"<{ntiles}{'Q' if off_elem_sz == 8 else 'I'}"
        cnt_fmt = f"<{ntiles}{'Q' if cnt_elem_sz == 8 else 'I'}"

        if off_tag >= base_off and off_tag + ntiles * off_elem_sz <= base_off + len(ifd_buf):
            b_off = ifd_buf[off_tag - base_off:off_tag - base_off + ntiles * off_elem_sz]
        else:
            b_off, _ = fetch_range(conn, path, off_tag, off_tag + ntiles * off_elem_sz - 1)

        if cnt_tag >= base_off and cnt_tag + ntiles * cnt_elem_sz <= base_off + len(ifd_buf):
            b_cnt = ifd_buf[cnt_tag - base_off:cnt_tag - base_off + ntiles * cnt_elem_sz]
        else:
            b_cnt, _ = fetch_range(conn, path, cnt_tag, cnt_tag + ntiles * cnt_elem_sz - 1)

        tile_offsets = struct.unpack(off_fmt, b_off)
        tile_counts = struct.unpack(cnt_fmt, b_cnt)
        return ntiles, comp_type, predictor, tile_offsets, tile_counts
    else:
        raise RuntimeError(f"Unknown TIFF magic: {magic}")


def list_year_keys(var, year):
    url = f"https://os.unil.cloud.switch.ch/chelsa02/?prefix=chelsa/global/daily/{var}/{year}/&max-keys=1000"
    req = urllib.request.Request(url, headers={"User-Agent": "TUBITAK-1002-GitHubActions/2.0"})
    xml_bytes = urllib.request.urlopen(req, timeout=25).read()
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
    for attempt in range(5):
        try:
            conn = get_conn() if attempt == 0 else reset_conn()
            ntiles, comp_type, predictor, tile_offsets, tile_counts = parse_ifd_and_tiles(conn, path)

            if ntiles == 435:
                s_top, e_top = tile_offsets[103], tile_offsets[103] + tile_counts[103] - 1
                s_bot, e_bot = tile_offsets[132], tile_offsets[132] + tile_counts[132] - 1
                b_top, _ = fetch_range(conn, path, s_top, e_top)
                b_bot, _ = fetch_range(conn, path, s_bot, e_bot)
                canvas_clt = np.empty((1024, 512), dtype=np.uint16)
                canvas_clt[0:512, :] = decompress_tile(b_top, comp_type, predictor)
                canvas_clt[512:1024, :] = decompress_tile(b_bot, comp_type, predictor)
                return (dt_str, canvas_clt[np.ix_(CLT_REL_ROWS, CLT_REL_COLS)].copy(), len(b_top) + len(b_bot))

            if ntiles == 20880:
                # Stripped TIFF (RowsPerStrip=1, 20880 scanlines of width 43200)
                s_str = min(tile_offsets[ROW_START:ROW_END])
                e_str = max(tile_offsets[i] + tile_counts[i] for i in range(ROW_START, ROW_END)) - 1
                b_str, _ = fetch_range(conn, path, s_str, e_str)
                cropped = np.empty((ROW_END - ROW_START, COL_END - COL_START), dtype=np.uint16)
                for r_i, s_idx in enumerate(range(ROW_START, ROW_END)):
                    rel_s = tile_offsets[s_idx] - s_str
                    comp_slice = b_str[rel_s:rel_s + tile_counts[s_idx]]
                    raw = zlib.decompress(comp_slice) if comp_type in (8, 32946) else (imagecodecs.lzw_decode(comp_slice) if comp_type == 5 else comp_slice)
                    row_arr = np.frombuffer(raw, dtype="<u2")
                    if predictor == 2:
                        row_arr = np.cumsum(row_arr, dtype=np.uint16)
                    cropped[r_i, :] = row_arr[COL_START:COL_END]
                return (dt_str, cropped, len(b_str))

            # Check if row 10 tiles (898, 899, 900) are contiguous and reasonably sized
            s10 = min(tile_offsets[898], tile_offsets[899], tile_offsets[900])
            e10 = max(tile_offsets[898] + tile_counts[898], tile_offsets[899] + tile_counts[899], tile_offsets[900] + tile_counts[900]) - 1
            s11 = min(tile_offsets[983], tile_offsets[984], tile_offsets[985])
            e11 = max(tile_offsets[983] + tile_counts[983], tile_offsets[984] + tile_counts[984], tile_offsets[985] + tile_counts[985]) - 1

            b10, _ = fetch_range(conn, path, s10, e10)
            b11, _ = fetch_range(conn, path, s11, e11)

            canvas = np.empty((1024, 1536), dtype=np.uint16)
            for tx_i, t_idx in enumerate([898, 899, 900]):
                rel_s = tile_offsets[t_idx] - s10
                comp_slice = b10[rel_s:rel_s + tile_counts[t_idx]]
                canvas[0:512, tx_i * 512:(tx_i + 1) * 512] = decompress_tile(comp_slice, comp_type, predictor)

            for tx_i, t_idx in enumerate([983, 984, 985]):
                rel_s = tile_offsets[t_idx] - s11
                comp_slice = b11[rel_s:rel_s + tile_counts[t_idx]]
                canvas[512:1024, tx_i * 512:(tx_i + 1) * 512] = decompress_tile(comp_slice, comp_type, predictor)

            return (dt_str, canvas[REL_R0:REL_R1, REL_C0:REL_C1].copy(), len(b10) + len(b11))
        except Exception as e:
            if attempt == 4:
                print(f"[GÜN HATASI - NODATA ATANDI] {key} sunucuda bozuk ({e}), 65535 (NaN) olarak işaretlendi.", flush=True)
                return (dt_str, np.full((ROW_END - ROW_START, COL_END - COL_START), 65535, dtype=np.uint16), 0)
            time.sleep(0.8 * (attempt + 1))


def main():
    var = sys.argv[1]
    y_start = int(sys.argv[2]) if len(sys.argv) > 2 else 1979
    y_end = int(sys.argv[3]) if len(sys.argv) > 3 else 2025
    release_tag = "chelsa-v2.1-data"

    outside_mask = None
    if os.path.exists("calisma_alani_statik_grid_1km.npz"):
        st = np.load("calisma_alani_statik_grid_1km.npz")
        outside_mask = (st["province_id"] == 0)

    existing_assets = set()
    res_check = subprocess.run(["gh", "release", "view", release_tag, "--json", "assets"], capture_output=True, text=True)
    if res_check.returncode == 0:
        for a in json.loads(res_check.stdout).get("assets", []):
            existing_assets.add(a["name"])
    else:
        subprocess.run(["gh", "release", "create", release_tag, "--title", "CHELSA-daily v2.1 Ege & Akdeniz 1km Küpleri (1979-2025)", "--notes", "Otomatik bulut indirme çıktıları."], check=False)

    failed_years = []
    for year in range(y_start, y_end + 1):
        if var == "pr" and year >= 2020:
            continue
        if var == "clt" and year == 2007:
            continue
        out_fname = f"CHELSA_{var}_{year}_EgeAkdeniz_1km.npz"
        if out_fname in existing_assets:
            print(f"[ATLANDI] {out_fname} zaten GitHub Release içinde mevcut.", flush=True)
            continue

        try:
            items = list_year_keys(var, year)
            if not items:
                continue

            t0 = time.time()
            with ThreadPoolExecutor(max_workers=24) as ex:
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

            subprocess.run(["gh", "release", "upload", release_tag, out_fname, "--clobber"], check=False)
            os.remove(out_fname)
        except Exception as e:
            print(f"[UYARI] {var} {year} hata aldı, sonraki yıla geçiliyor: {e}", flush=True)
            failed_years.append(year)

    if failed_years:
        print(f"Tamamlanamayan yıllar ({var}): {failed_years}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
