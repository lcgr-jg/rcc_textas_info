"""
claude_vision_extractor.py — Tier 3 fallback using Claude API vision
=====================================================================
For scanned PDFs that Tesseract can't reliably extract (balance pass rate <25%),
send the rasterized page image to Claude's API for structured extraction.

Accuracy: near-perfect (Claude understands table context, not just pixel patterns)
Cost: ~$0.01-0.03 per page (Sonnet) or ~$0.05-0.15 per page (Opus)
Latency: ~5-15 seconds per page

Requirements:
    pip install anthropic PyMuPDF Pillow

Setup:
    export ANTHROPIC_API_KEY="your-key-here"
    # Or set in code: os.environ["ANTHROPIC_API_KEY"] = "your-key"

Usage:
    from claude_vision_extractor import extract_with_claude_vision
    result = extract_with_claude_vision("path/to/scanned.pdf")
"""

import os
import re
import json
import base64
import io
import tempfile
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image

# Try PyMuPDF first (Windows), fall back to pdftoppm (Linux)
try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


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

EXTRACTION_PROMPT = """You are extracting data from a Texas Railroad Commission Form R-1 
(Monthly Report and Operations Statement for Refineries).

Look at this scanned page and extract the material balance table. 
Return ONLY a JSON object with this exact structure — no markdown, no backticks, no explanation:

{
  "operator": "operator name from header",
  "month": "Month Year from header",
  "registration_no": "XX-XXXX from header",
  "refinery_name": "refinery name if visible",
  "rows": [
    {
      "code": 1,
      "storage_bom": 0,
      "receipts": 0,
      "input": 0,
      "products_mfg": 0,
      "fuel_used": 0,
      "deliveries": 0,
      "storage_eom": 0
    }
  ]
}

Rules:
- Include ALL 24 product codes (1-24), even if a row is blank (use 0 for blank cells)
- Numbers are in barrels — use integers, no commas
- If a cell contains a dash (-) or is empty, use 0
- The "code" field must be the integer 1-24 printed in the Code column
- Read the numbers carefully — this is a scanned document and precision matters
- Return ONLY valid JSON, nothing else"""


def rasterize_page_to_bytes(pdf_path: str, dpi: int = 200) -> bytes:
    """Rasterize page 1 to JPEG bytes for the API."""
    if HAS_FITZ:
        doc = fitz.open(pdf_path)
        page = doc[0]
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        doc.close()
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, 'page')
            subprocess.run(
                ['pdftoppm', '-jpeg', '-r', str(dpi), '-f', '1', '-l', '1', pdf_path, prefix],
                check=True, capture_output=True
            )
            img_path = sorted(Path(tmpdir).glob('page-*.jpg'))[0]
            img = Image.open(img_path)
    
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return buf.getvalue()


def extract_with_claude_vision(
    pdf_path: str,
    model: str = "claude-sonnet-4-5-20250514",
    dpi: int = 200,
    api_key: str = None,
) -> dict:
    """
    Extract Form R-1 data by sending the page image to Claude's vision API.
    
    Args:
        pdf_path: Path to the scanned PDF
        model: Claude model to use. Sonnet is best cost/accuracy tradeoff.
               Use "claude-opus-4-6" for maximum accuracy on very difficult scans.
        dpi: Resolution for rasterization (200 is usually sufficient)
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
    
    Returns:
        dict with header, data, mode, tier, error, balance_stats, cost_estimate
    """
    if not HAS_ANTHROPIC:
        return {
            'header': {}, 'data': [], 'mode': 'claude_vision', 'tier': 3,
            'error': 'anthropic package not installed. Run: pip install anthropic'
        }
    
    # Rasterize
    try:
        img_bytes = rasterize_page_to_bytes(pdf_path, dpi=dpi)
    except Exception as e:
        return {
            'header': {}, 'data': [], 'mode': 'claude_vision', 'tier': 3,
            'error': f'Rasterize failed: {e}'
        }
    
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # Call Claude API
    client = Anthropic(api_key=api_key) if api_key else Anthropic()
    
    try:
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        }
                    },
                    {
                        "type": "text",
                        "text": EXTRACTION_PROMPT,
                    }
                ]
            }]
        )
    except Exception as e:
        return {
            'header': {}, 'data': [], 'mode': 'claude_vision', 'tier': 3,
            'error': f'API call failed: {e}'
        }
    
    # Parse response
    response_text = message.content[0].text.strip()
    
    # Clean up potential markdown wrapping
    response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
    response_text = re.sub(r'\s*```$', '', response_text)
    
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as e:
        return {
            'header': {}, 'data': [], 'mode': 'claude_vision', 'tier': 3,
            'error': f'JSON parse failed: {e}\nRaw response: {response_text[:500]}'
        }
    
    # Build structured output
    header = {
        'operator': parsed.get('operator', ''),
        'month_str': parsed.get('month', ''),
        'registration_no': parsed.get('registration_no', ''),
        'refinery_name': parsed.get('refinery_name', ''),
    }
    
    rows = []
    for r in parsed.get('rows', []):
        code = int(r.get('code', 0))
        if code < 1 or code > 24:
            continue
        rows.append({
            'Code': code,
            'Storage Beginning of Month': float(r.get('storage_bom', 0)),
            'Receipts': float(r.get('receipts', 0)),
            'Input': float(r.get('input', 0)),
            'Products Manufactured': float(r.get('products_mfg', 0)),
            'Fuel Used': float(r.get('fuel_used', 0)),
            'Deliveries': float(r.get('deliveries', 0)),
            'Storage End of Month': float(r.get('storage_eom', 0)),
        })
    
    # Fill missing codes
    existing = {r['Code'] for r in rows}
    for code in range(1, 25):
        if code not in existing:
            rows.append({**{c: 0.0 for c in DATA_COLUMNS}, 'Code': code})
    rows.sort(key=lambda r: r['Code'])
    
    # Validate
    df = pd.DataFrame(rows)
    inflows = df['Storage Beginning of Month'] + df['Receipts'] + df['Products Manufactured']
    outflows = df['Input'] + df['Fuel Used'] + df['Deliveries'] + df['Storage End of Month']
    df['balance_diff'] = inflows - outflows
    df['balanced'] = df['balance_diff'].abs() < 2
    
    active = df[df[DATA_COLUMNS].sum(axis=1) > 0]
    n_active = len(active)
    n_balanced = int(active['balanced'].sum()) if n_active > 0 else 0
    pass_rate = n_balanced / n_active if n_active > 0 else 1.0
    
    # Estimate cost from usage
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    
    # Approximate pricing (Sonnet 4.5: $3/$15 per MTok)
    cost = (input_tokens * 3 + output_tokens * 15) / 1_000_000
    
    return {
        'header': header,
        'data': rows,
        'mode': 'claude_vision',
        'tier': 1 if pass_rate >= 0.75 else 2,
        'error': None,
        'balance_stats': {
            'total_active': n_active,
            'balanced_count': n_balanced,
            'pass_rate': round(pass_rate, 3),
        },
        'api_usage': {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'estimated_cost_usd': round(cost, 4),
        },
    }


# ── Integration helper for the main pipeline ──

def extract_with_fallback(
    pdf_path: str,
    tier2_result: dict = None,
    force_claude: bool = False,
    model: str = "claude-sonnet-4-5-20250514",
) -> dict:
    """
    Decide whether to use Tier 2 (Tesseract) or Tier 3 (Claude vision).
    
    Args:
        pdf_path: Path to PDF
        tier2_result: Result from scanned_extractor.extract_scanned_adaptive()
                      If None, skips Tier 2 check.
        force_claude: Always use Claude API (skip Tier 2)
        model: Claude model for vision extraction
    
    Returns:
        Best available extraction result
    """
    # If Tier 2 was good enough, keep it
    if tier2_result and not force_claude:
        stats = tier2_result.get('balance_stats', {})
        if stats.get('pass_rate', 0) >= 0.5:
            return tier2_result
    
    # Try Claude vision
    result = extract_with_claude_vision(pdf_path, model=model)
    
    if result['error']:
        # Claude failed — return Tier 2 if available, otherwise error
        if tier2_result:
            tier2_result['note'] = f'Claude API fallback failed: {result["error"]}'
            return tier2_result
        return result
    
    return result
