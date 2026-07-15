"""
r1_extractor.py — Unified Form R-1 extraction with tiered fallback
===================================================================
Routes each PDF through the optimal extraction path:

  Tier 1: pdfplumber (digital PDFs) — fast, ~100% accurate
  Tier 2: Tesseract OCR with auto grid detection (clean scans) — ~60-75% accuracy
  Tier 3: Claude API vision (difficult scans) — ~95%+ accuracy, ~$0.01-0.03/page

Usage:
    from r1_extractor import extract_r1
    result = extract_r1("path/to/form.pdf", use_claude_fallback=True)

Requirements:
    pip install pdfplumber pytesseract Pillow numpy scipy pandas PyMuPDF anthropic
"""

import os
import re
import pandas as pd
import pdfplumber

from scanned_extractor import (
    extract_scanned_adaptive, rasterize_page, parse_number, 
    PRODUCT_CODES, DATA_COLUMNS
)

# Optional: Claude vision fallback
try:
    from claude_vision_extractor import extract_with_claude_vision
    HAS_CLAUDE = True
except ImportError:
    HAS_CLAUDE = False


def is_digital_pdf(pdf_path: str) -> bool:
    """Check if PDF has extractable text (digital vs scanned)."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = pdf.pages[0].extract_text() or ''
            return len(text.strip()) > 200
    except Exception:
        return False


def extract_digital(pdf_path: str) -> dict:
    """Tier 1: Extract from digital PDF via pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        page1_text = pdf.pages[0].extract_text() or ''
        
        header = {}
        m = re.search(r'Operator[:\s]+(.+?)(?:Month|$)', page1_text, re.IGNORECASE)
        if m: header['operator'] = m.group(1).strip()
        m = re.search(r'(January|February|March|April|May|June|July|August|'
                       r'September|October|November|December)\s+\d{4}', page1_text, re.IGNORECASE)
        if m: header['month_str'] = m.group(0)
        m = re.search(r'(\d{2}-\d{4})', page1_text)
        if m: header['registration_no'] = m.group(1)
        m = re.search(r'Refinery\s*(?:Name)?\s*[:.]?\s*(.+?)(?:\n|$)', page1_text, re.IGNORECASE)
        if m: header['refinery_name'] = m.group(1).strip()
        
        tables = pdf.pages[0].extract_tables()
        if not tables:
            return {'header': header, 'data': [], 'mode': 'digital', 'tier': 3,
                    'error': 'No table found on page 1'}
        
        t = tables[0]
        rows = []
        for row in t[1:]:
            if row[0] and 'TOTALS' in str(row[0]):
                continue
            code_val = parse_number(row[1])
            if code_val == 0:
                continue
            code = int(code_val)
            rows.append({
                'Code': code,
                'Storage Beginning of Month': parse_number(row[2]),
                'Receipts': parse_number(row[3]),
                'Input': parse_number(row[4]),
                'Products Manufactured': parse_number(row[5]),
                'Fuel Used': parse_number(row[6]),
                'Deliveries': parse_number(row[7]),
                'Storage End of Month': parse_number(row[8]),
            })
        
        # Validate
        df = pd.DataFrame(rows)
        if not df.empty:
            inflows = df['Storage Beginning of Month'] + df['Receipts'] + df['Products Manufactured']
            outflows = df['Input'] + df['Fuel Used'] + df['Deliveries'] + df['Storage End of Month']
            active = df[df[DATA_COLUMNS].sum(axis=1) > 0]
            n_active = len(active)
            if n_active > 0:
                balanced = ((inflows - outflows).abs() < 2)
                n_balanced = int(active.index.isin(df[balanced].index).sum())
                pass_rate = n_balanced / n_active
            else:
                n_balanced, pass_rate = 0, 1.0
        else:
            n_active, n_balanced, pass_rate = 0, 0, 1.0
        
        return {
            'header': header,
            'data': rows,
            'mode': 'digital',
            'tier': 1,
            'error': None,
            'balance_stats': {
                'total_active': n_active,
                'balanced_count': n_balanced,
                'pass_rate': round(pass_rate, 3),
            }
        }


def extract_r1(
    pdf_path: str,
    use_claude_fallback: bool = True,
    claude_model: str = "claude-sonnet-4-5-20250514",
    claude_threshold: float = 0.25,
) -> dict:
    """
    Extract Form R-1 data using the best available method.
    
    Pipeline:
        1. Try pdfplumber (digital PDFs)
        2. If scanned → Tesseract OCR with auto grid detection
        3. If Tesseract balance rate < threshold → Claude API vision
    
    Args:
        pdf_path: Path to the Form R-1 PDF
        use_claude_fallback: Enable Tier 3 Claude API fallback
        claude_model: Which Claude model to use for vision extraction
        claude_threshold: Balance pass rate below which to escalate to Claude
    
    Returns:
        dict with: header, data, mode, tier, error, balance_stats
        mode is one of: 'digital', 'scanned_ocr', 'claude_vision'
        tier is 1 (high confidence), 2 (partial), or 3 (needs review)
    """
    # ── Tier 1: Digital PDF ──
    if is_digital_pdf(pdf_path):
        result = extract_digital(pdf_path)
        if result['data']:
            return result
    
    # ── Tier 2: Tesseract OCR ──
    result = extract_scanned_adaptive(pdf_path)
    
    if result['error']:
        tier2_result = result
    else:
        stats = result.get('balance_stats', {})
        pass_rate = stats.get('pass_rate', 0)
        
        # If good enough, return Tier 2
        if pass_rate >= claude_threshold:
            return result
        
        tier2_result = result
    
    # ── Tier 3: Claude API Vision ──
    if use_claude_fallback and HAS_CLAUDE:
        claude_result = extract_with_claude_vision(pdf_path, model=claude_model)
        
        if not claude_result['error']:
            return claude_result
        else:
            # Claude failed — return Tier 2 with a note
            if tier2_result and tier2_result.get('data'):
                tier2_result['note'] = f'Claude fallback failed: {claude_result["error"]}'
                return tier2_result
    
    # No Claude available or both failed
    if tier2_result and tier2_result.get('data'):
        tier2_result['note'] = 'Below confidence threshold but Claude fallback not available'
        return tier2_result
    
    return {
        'header': tier2_result.get('header', {}) if tier2_result else {},
        'data': [],
        'mode': 'failed',
        'tier': 3,
        'error': 'All extraction methods failed',
    }


def extract_batch(
    pdf_paths: list,
    use_claude_fallback: bool = True,
    claude_model: str = "claude-sonnet-4-5-20250514",
) -> tuple:
    """
    Extract a batch of PDFs with progress reporting.
    
    Returns:
        (results_df, log_df) — extracted data and processing log
    """
    all_rows = []
    log_entries = []
    
    for i, pdf_path in enumerate(pdf_paths):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(pdf_paths)}] Processing...")
        
        result = extract_r1(
            pdf_path,
            use_claude_fallback=use_claude_fallback,
            claude_model=claude_model,
        )
        
        # Parse facility ID and date from filename
        fname = Path(pdf_path).stem
        parts = fname.split('___')
        id_part = str(parts[1].split('_')[0]) if len(parts) >= 2 else 'unknown'
        
        try:
            date_part = parts[0]
            year, month = date_part.split('_')[:2]
            year, month = int(year), int(month)
            from datetime import datetime
            if month == 12:
                date = datetime(year, 12, 31)
            else:
                date = datetime(year, month + 1, 1) - pd.Timedelta(days=1)
        except (ValueError, IndexError):
            date = None
        
        log_entries.append({
            'file': pdf_path,
            'id': id_part,
            'date': date,
            'mode': result['mode'],
            'tier': result.get('tier', 3),
            'error': result.get('error'),
            'active_rows': result.get('balance_stats', {}).get('total_active', 0),
            'balanced_rows': result.get('balance_stats', {}).get('balanced_count', 0),
            'pass_rate': result.get('balance_stats', {}).get('pass_rate', 0),
            'api_cost': result.get('api_usage', {}).get('estimated_cost_usd', 0),
        })
        
        for row in result.get('data', []):
            row['id'] = id_part
            row['date'] = date
            row['extraction_mode'] = result['mode']
            row['extraction_tier'] = result.get('tier', 3)
            all_rows.append(row)
    
    results_df = pd.DataFrame(all_rows)
    log_df = pd.DataFrame(log_entries)
    
    # Summary
    if not log_df.empty:
        print(f"\n{'='*60}")
        print(f"BATCH EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total PDFs: {len(log_df)}")
        print(f"By mode:")
        for mode, count in log_df['mode'].value_counts().items():
            print(f"  {mode}: {count}")
        print(f"By tier:")
        for tier, count in log_df['tier'].value_counts().sort_index().items():
            print(f"  Tier {tier}: {count}")
        errors = log_df[log_df['error'].notna()]
        if not errors.empty:
            print(f"Errors: {len(errors)}")
        total_cost = log_df['api_cost'].sum()
        if total_cost > 0:
            print(f"Claude API cost: ${total_cost:.2f}")
    
    return results_df, log_df
