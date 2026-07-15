"""
scanned_extractor.py — Adaptive OCR extraction for scanned Form R-1 PDFs
=========================================================================
Auto-detects the table grid on any scanned R-1, extracts cell-by-cell,
validates with the balance equation, and assigns a confidence tier.

Requires: pytesseract, Pillow, numpy, scipy, pandas
          System: tesseract-ocr
          For rasterization: PyMuPDF (fitz) on Windows, or poppler on Linux

Usage:
    from scanned_extractor import extract_scanned_adaptive
    result = extract_scanned_adaptive('path/to/scanned.pdf')
"""

import os
import re
import subprocess
import tempfile
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from scipy.signal import find_peaks

# Try PyMuPDF first (works on Windows), fall back to pdftoppm
try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


PRODUCT_CODES = {
    1:'Propane',2:'Butane',3:'Butane-Propane',4:'Motor Gasoline',
    5:'Kerosene',6:'Home Heating Oil',7:'Diesel Fuel',
    8:'Other Middle Distillates',9:'Aviation Gasoline',
    10:'Kerosene-Type Jet Fuel',11:'Naphtha-Type Jet Fuel',
    12:'Fuel Oils #4 For Utility Use',13:'Fuel Oils #5,#6 For Utility Use',
    14:'Fuel Oils #4 For Non-Utility Use',15:'Fuel Oils #5,#6 For Non-Utility Use',
    16:'Bunker C',17:'Navy Special',18:'Other Residual Fuels',
    19:'Petrochemical Feedstock',20:'Lubricants',21:'Special Naphthas',
    22:'Solvent Products',23:'Miscellaneous',24:'Crude Oil',
}

DATA_COLUMNS = [
    'Storage Beginning of Month', 'Receipts', 'Input',
    'Products Manufactured', 'Fuel Used', 'Deliveries', 'Storage End of Month'
]


# ── Rasterization ──

def rasterize_page(pdf_path: str, dpi: int = 200) -> Image.Image:
    """Rasterize page 1 of a PDF to a PIL Image. Cross-platform."""
    if HAS_FITZ:
        doc = fitz.open(pdf_path)
        page = doc[0]
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, 'page')
            subprocess.run(
                ['pdftoppm', '-jpeg', '-r', str(dpi), '-f', '1', '-l', '1', pdf_path, prefix],
                check=True, capture_output=True
            )
            img_path = sorted(Path(tmpdir).glob('page-*.jpg'))[0]
            return Image.open(img_path).copy()


# ── Grid Detection ──

def detect_table_grid(img: Image.Image) -> tuple:
    """
    Auto-detect the material balance table grid from any scanned Form R-1.
    
    Returns:
        (v_lines, h_lines) — lists of pixel coordinates for vertical and horizontal grid lines
        Returns (None, None) if detection fails.
    """
    arr = np.array(img.convert('L'))
    threshold = 100
    
    # Horizontal lines
    row_dark = (arr < threshold).sum(axis=1)
    hpeaks, _ = find_peaks(row_dark, height=arr.shape[1] * 0.15, distance=12)
    
    # Filter to table area (middle 75% of page)
    y_min = int(arr.shape[0] * 0.15)
    y_max = int(arr.shape[0] * 0.9)
    hpeaks = hpeaks[(hpeaks >= y_min) & (hpeaks <= y_max)]
    
    if len(hpeaks) < 10:
        return None, None
    
    # Find data-row-like spacings (18-50px at 200 DPI)
    spacings = np.diff(hpeaks)
    data_mask = (spacings >= 18) & (spacings <= 50)
    data_indices = np.where(data_mask)[0]
    
    if len(data_indices) < 5:
        return None, None
    
    # Extend to capture full table
    start_idx = max(0, data_indices[0] - 2)
    end_idx = min(len(hpeaks) - 1, data_indices[-1] + 2)
    table_h = hpeaks[start_idx:end_idx + 1].tolist()
    
    # Vertical lines spanning the table region
    table_region = arr[table_h[0]:table_h[-1], :]
    col_dark = (table_region < threshold).sum(axis=0)
    min_height = len(table_region) * 0.25
    vpeaks, _ = find_peaks(col_dark, height=min_height, distance=25)
    
    return vpeaks.tolist(), table_h


def identify_data_columns(v_lines: list) -> list:
    """
    From detected vertical lines, identify the 7 data column boundaries.
    
    Logic: The 7 data columns are the rightmost wide (>100px) gaps,
    excluding the leftmost wide gap (Name of Material).
    """
    if len(v_lines) < 4:
        return []
    
    gaps = [(v_lines[i], v_lines[i+1], v_lines[i+1] - v_lines[i]) 
            for i in range(len(v_lines) - 1)]
    
    # Wide gaps (>100px) are real columns
    wide = [(x1, x2, w) for x1, x2, w in gaps if w > 100]
    
    if len(wide) >= 8:
        # First wide gap = Material Name, next 7 = data columns
        data = [(x1, x2) for x1, x2, w in wide[1:8]]
    elif len(wide) == 7:
        data = [(x1, x2) for x1, x2, w in wide]
    else:
        # Fallback: take rightmost 7 gaps
        data = [(x1, x2) for x1, x2, w in gaps[-7:]]
    
    return data


def find_data_start_row(h_lines: list, v_lines: list, img: Image.Image) -> int:
    """
    Find which h-line index starts the first data row (Code 1 = Propane).
    
    Strategy: OCR the Code column in each row, look for "1".
    Fallback: skip rows until we find a row with height 20-45px.
    """
    if not v_lines or len(v_lines) < 4:
        return 0
    
    # Find the Code column: narrow gap (40-80px) after the widest left gap
    gaps = [(v_lines[i], v_lines[i+1], v_lines[i+1] - v_lines[i], i) 
            for i in range(len(v_lines) - 1)]
    
    # The Code column is the first narrow gap after Material Name
    # Material Name is the widest gap in the left half
    left_half = [g for g in gaps if g[0] < img.width * 0.4]
    if not left_half:
        return 0
    
    mat_gap = max(left_half, key=lambda g: g[2])
    mat_idx = mat_gap[3]
    
    # Code column is right after material name, typically 40-80px wide
    code_x1 = v_lines[mat_idx + 1]
    code_x2 = v_lines[min(mat_idx + 2, len(v_lines) - 1)]
    
    # Scan rows for Code = "1"
    SCALE = 4
    for i in range(min(15, len(h_lines) - 1)):
        y1 = h_lines[i] + 2
        y2 = h_lines[i + 1] - 2
        if y2 - y1 < 8:
            continue
        
        cell = img.crop((code_x1, y1, code_x2, y2))
        w, h = cell.size
        if w < 5 or h < 5:
            continue
        cell = cell.resize((w * SCALE, h * SCALE), Image.LANCZOS)
        text = pytesseract.image_to_string(
            cell, config='--psm 7 -c tessedit_char_whitelist=0123456789'
        ).strip()
        
        if text == '1':
            return i
    
    # Fallback: find first "normal height" row after a taller row
    spacings = [h_lines[i+1] - h_lines[i] for i in range(len(h_lines) - 1)]
    for i in range(1, len(spacings)):
        if spacings[i] < 45 and spacings[i-1] >= 45:
            return i
    
    # Last resort: skip 2 rows (header area)
    return min(2, len(h_lines) - 25)


# ── Number Parsing ──

def parse_number(raw) -> float:
    """Parse a number from OCR text. Returns 0.0 for blanks/dashes."""
    if raw is None:
        return 0.0
    s = str(raw).strip().replace(',', '').replace(' ', '').replace('|', '')
    s = re.sub(r'[^0-9.\-]', '', s)
    if not s or s == '-':
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


# ── Header Extraction ──

def extract_header(img: Image.Image) -> dict:
    """OCR the top of the page for operator, month, registration number."""
    header_region = img.crop((0, 0, img.width, int(img.height * 0.25)))
    text = pytesseract.image_to_string(header_region)
    
    header = {}
    m = re.search(
        r'(January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+\d{4}', text, re.IGNORECASE
    )
    if m:
        header['month_str'] = m.group(0)
    
    m = re.search(r'(\d{2}-\d{4})', text)
    if m:
        header['registration_no'] = m.group(1)
    
    m = re.search(r'Operator[:\s]+(.+?)(?:Month|Address|$)', text, re.IGNORECASE)
    if m:
        header['operator'] = m.group(1).strip()
    
    m = re.search(r'Refinery\s*(?:Name)?\s*[:.]?\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m:
        header['refinery_name'] = m.group(1).strip()
    
    return header


# ── Main Extraction ──

def extract_scanned_adaptive(pdf_path: str, dpi: int = 200) -> dict:
    """
    Extract Form R-1 data from a scanned PDF with auto grid detection.
    
    Returns dict with:
        header: operator, month, registration_no
        data: list of row dicts (Code + 7 value columns)
        mode: 'scanned_ocr'
        tier: 1 (>75% balanced), 2 (25-75%), 3 (<25% — needs manual review)
        error: error message or None
        balance_stats: {total_active, balanced_count, pass_rate}
    """
    try:
        img = rasterize_page(pdf_path, dpi=dpi)
    except Exception as e:
        return {'header': {}, 'data': [], 'mode': 'scanned_ocr', 'tier': 3,
                'error': f'Rasterize failed: {e}'}
    
    header = extract_header(img)
    
    # Grid detection
    v_lines, h_lines = detect_table_grid(img)
    if not v_lines or not h_lines:
        return {'header': header, 'data': [], 'mode': 'scanned_ocr', 'tier': 3,
                'error': 'Could not detect table grid'}
    
    # Column identification
    col_bounds = identify_data_columns(v_lines)
    if len(col_bounds) < 7:
        return {'header': header, 'data': [], 'mode': 'scanned_ocr', 'tier': 3,
                'error': f'Only {len(col_bounds)} columns detected (need 7)'}
    
    # Find data start
    data_start = find_data_start_row(h_lines, v_lines, img)
    
    # Cell-by-cell extraction
    PAD = 3
    SCALE = 4
    rows = []
    
    for code in range(1, 25):
        h_idx = data_start + (code - 1)
        if h_idx + 1 >= len(h_lines):
            break
        
        y_top = h_lines[h_idx] + PAD
        y_bot = h_lines[h_idx + 1] - PAD
        if y_bot <= y_top:
            continue
        
        row_data = {'Code': code}
        
        for col_i, col_name in enumerate(DATA_COLUMNS):
            if col_i >= len(col_bounds):
                row_data[col_name] = 0.0
                continue
            
            x1, x2 = col_bounds[col_i]
            cell = img.crop((x1 + PAD, y_top, x2 - PAD, y_bot))
            w, h = cell.size
            
            if w < 5 or h < 5:
                row_data[col_name] = 0.0
                continue
            
            # Scale up for better Tesseract accuracy
            cell = cell.resize((w * SCALE, h * SCALE), Image.LANCZOS)
            text = pytesseract.image_to_string(
                cell, config='--psm 7 -c tessedit_char_whitelist=0123456789,.-'
            ).strip()
            row_data[col_name] = parse_number(text)
        
        rows.append(row_data)
    
    # Fill missing codes with zeros
    existing_codes = {r['Code'] for r in rows}
    for code in range(1, 25):
        if code not in existing_codes:
            rows.append({**{c: 0.0 for c in DATA_COLUMNS}, 'Code': code})
    rows.sort(key=lambda r: r['Code'])
    
    # Validate and assign tier
    df = pd.DataFrame(rows)
    inflows = df['Storage Beginning of Month'] + df['Receipts'] + df['Products Manufactured']
    outflows = df['Input'] + df['Fuel Used'] + df['Deliveries'] + df['Storage End of Month']
    df['balance_diff'] = inflows - outflows
    df['balanced'] = df['balance_diff'].abs() < 2
    
    active = df[df[DATA_COLUMNS].sum(axis=1) > 0]
    n_active = len(active)
    n_balanced = int(active['balanced'].sum()) if n_active > 0 else 0
    pass_rate = n_balanced / n_active if n_active > 0 else 1.0
    
    if pass_rate >= 0.75:
        tier = 1
    elif pass_rate >= 0.25:
        tier = 2
    else:
        tier = 3
    
    return {
        'header': header,
        'data': rows,
        'mode': 'scanned_ocr',
        'tier': tier,
        'error': None,
        'balance_stats': {
            'total_active': n_active,
            'balanced_count': n_balanced,
            'pass_rate': round(pass_rate, 3),
        },
    }
