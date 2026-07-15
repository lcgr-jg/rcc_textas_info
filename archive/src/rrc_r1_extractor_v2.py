"""
RRC Form R-1 Extractor — Dual-Mode Pipeline
=============================================
Handles BOTH types of Form R-1 PDFs filed with the Texas Railroad Commission:

  1. Digital/native PDFs (e.g. Blanchard Refining) — extracted via pdfplumber
     with near-perfect accuracy, no OCR needed.
  2. Scanned/image PDFs (e.g. ExxonMobil Beaumont) — extracted via Tesseract
     OCR with positional column mapping. Accuracy ~85-90%, needs validation.

All processing is 100% local — no data leaves your machine.

Requirements:
    pip install pdfplumber pytesseract Pillow pandas pypdf
    # System packages:
    # sudo apt install tesseract-ocr poppler-utils   (Linux)
    # brew install tesseract poppler                  (macOS)

Usage:
    python rrc_r1_extractor.py <pdf_or_directory>
    python rrc_r1_extractor.py /path/to/all_filings/
"""

import sys
import os
import subprocess
import tempfile
import re
from pathlib import Path

import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd


# ─── Constants ────────────────────────────────────────────────────────────────

PRODUCT_CODES = {
    1:  "Propane",
    2:  "Butane",
    3:  "Butane-Propane",
    4:  "Motor Gasoline",
    5:  "Kerosene",
    6:  "Home Heating Oil",
    7:  "Diesel Fuel",
    8:  "Other Middle Distillates",
    9:  "Aviation Gasoline",
    10: "Kerosene-Type Jet Fuel",
    11: "Naphtha-Type Jet Fuel",
    12: "Fuel Oils #4 For Utility Use",
    13: "Fuel Oils #5,#6 For Utility Use",
    14: "Fuel Oils #4 For Non-Utility Use",
    15: "Fuel Oils #5,#6 For Non-Utility Use",
    16: "Bunker C",
    17: "Navy Special",
    18: "Other Residual Fuels",
    19: "Petrochemical Feedstock",
    20: "Lubricants",
    21: "Special Naphthas",
    22: "Solvent Products",
    23: "Miscellaneous",
    24: "Crude Oil",
}

BALANCE_COLUMNS = [
    "code", "material_name", "storage_bom", "receipts",
    "input_run_blends", "products_mfg", "fuel_used",
    "deliveries", "storage_eom"
]

CAPACITY_COLUMNS = ["code", "material_name", "capacity", "storage_eom"]


# ─── Detection: is this a digital or scanned PDF? ────────────────────────────

def is_digital_pdf(pdf_path: str) -> bool:
    """
    Check if a PDF has extractable text (digital) or is image-only (scanned).
    Heuristic: if page 1 yields >200 chars of text, it's digital.
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[0].extract_text() or ""
        return len(text.strip()) > 200


# ─── Parsing helpers ──────────────────────────────────────────────────────────

def parse_number(raw: str) -> float | None:
    """Clean a number string from either OCR or pdfplumber output."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s in ("-", "=", ":", "", "E", "R", "a", "s", ">", "®", "Fz"):
        return None
    s = s.replace(",", "").replace(" ", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_header_text(text: str) -> dict:
    """Parse header fields from page 1 text."""
    info = {}
    
    # Operator
    m = re.search(r"Operator[:\s]+(.+?)(?:Month|$)", text, re.IGNORECASE)
    if m:
        info["operator"] = m.group(1).strip().rstrip(".")
    
    # Month
    m = re.search(r"Month[:\s]+((?:January|February|March|April|May|June|July|August|"
                   r"September|October|November|December)\s+\d{4})", text, re.IGNORECASE)
    if m:
        info["month"] = m.group(1).strip()
    
    # Registration No
    m = re.search(r"Registration\s*No\.?\s*[:.]?\s*(\d{2}-\d{4})", text, re.IGNORECASE)
    if m:
        info["registration_no"] = m.group(1)
    
    # Refinery name/location
    m = re.search(r"Refinery\s*(?:Name)?\s*[:.]?\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        info["refinery_name"] = m.group(1).strip()
    
    # Address
    m = re.search(r"Address[:\s]+(.+?)(?:Registration|$)", text, re.IGNORECASE)
    if m:
        info["address"] = m.group(1).strip()
    
    return info


# ─── Mode 1: Digital PDF extraction ──────────────────────────────────────────

def extract_digital(pdf_path: str) -> dict:
    """Extract from a digitally-generated PDF using pdfplumber table detection."""
    with pdfplumber.open(pdf_path) as pdf:
        # --- Header ---
        page1_text = pdf.pages[0].extract_text() or ""
        header = extract_header_text(page1_text)
        
        # --- Page 1: Material Balance ---
        tables = pdf.pages[0].extract_tables()
        balance_df = pd.DataFrame()
        
        if tables:
            t = tables[0]
            # First row is headers, rest is data
            rows = []
            for row in t[1:]:  # skip header
                if row[0] and "TOTALS" in str(row[0]):
                    # This is the totals row — save separately
                    continue
                code = parse_number(row[1])
                if code is None:
                    continue
                code = int(code)
                rows.append({
                    "code": code,
                    "material_name": PRODUCT_CODES.get(code, f"Unknown ({code})"),
                    "storage_bom": parse_number(row[2]),
                    "receipts": parse_number(row[3]),
                    "input_run_blends": parse_number(row[4]),
                    "products_mfg": parse_number(row[5]),
                    "fuel_used": parse_number(row[6]),
                    "deliveries": parse_number(row[7]),
                    "storage_eom": parse_number(row[8]),
                })
            balance_df = pd.DataFrame(rows)
        
        # --- Last page: Capacity ---
        capacity_df = pd.DataFrame()
        last_page = pdf.pages[-1]
        cap_tables = last_page.extract_tables()
        if cap_tables:
            ct = cap_tables[0]
            cap_rows = []
            for row in ct[1:]:
                if row[0] and "TOTALS" in str(row[0]):
                    continue
                code = parse_number(row[2])
                if code is None:
                    continue
                code = int(code)
                cap_rows.append({
                    "code": code,
                    "material_name": PRODUCT_CODES.get(code, f"Unknown ({code})"),
                    "capacity": parse_number(row[1]),
                    "storage_eom": parse_number(row[3]),
                })
            capacity_df = pd.DataFrame(cap_rows)
        
        # --- Receipts & Deliveries (counterparty detail) ---
        # These span multiple pages. We extract the text which contains
        # counterparty names + code/barrel pairs embedded in the page text.
        receipts_text = []
        deliveries_text = []
        mode = None
        for page in pdf.pages[1:]:
            text = page.extract_text() or ""
            if "ACTUAL RECEIPTS" in text:
                mode = "receipts"
            elif "ACTUAL DELIVERIES" in text:
                mode = "deliveries"
            
            if mode == "receipts":
                receipts_text.append(text)
            elif mode == "deliveries":
                deliveries_text.append(text)
        
        return {
            "file": os.path.basename(pdf_path),
            "extraction_mode": "digital",
            "header": header,
            "material_balance": balance_df,
            "capacity": capacity_df,
            "page_count": len(pdf.pages),
            "receipts_raw": "\n".join(receipts_text),
            "deliveries_raw": "\n".join(deliveries_text),
        }


# ─── Mode 2: Scanned/OCR PDF extraction ──────────────────────────────────────

# Pixel layout for scanned R-1 at 200 DPI
OCR_TABLE_Y_START = 330
OCR_TABLE_Y_END = 870
OCR_ROW_HEIGHT = 28
OCR_COLUMNS = [
    ("material_name",    0,    520),
    ("code",             520,  610),
    ("storage_bom",      610,  790),
    ("receipts",         790,  1000),
    ("input_run_blends", 1000, 1220),
    ("products_mfg",     1220, 1470),
    ("fuel_used",        1470, 1700),
    ("deliveries",       1700, 1920),
    ("storage_eom",      1920, 2200),
]


def rasterize_pdf(pdf_path: str, dpi: int = 200) -> list[str]:
    """Convert PDF pages to JPEG images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "page")
        subprocess.run(
            ["pdftoppm", "-jpeg", "-r", str(dpi), pdf_path, prefix],
            check=True, capture_output=True
        )
        pages = sorted(Path(tmpdir).glob("page-*.jpg"))
        result = []
        for p in pages:
            dest = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            dest.write(p.read_bytes())
            dest.close()
            result.append(dest.name)
    return result


def extract_scanned(pdf_path: str) -> dict:
    """Extract from a scanned/image PDF using Tesseract OCR."""
    pages = rasterize_pdf(pdf_path, dpi=200)
    
    try:
        img1 = Image.open(pages[0])
        
        # --- Header (OCR top of page) ---
        header_crop = img1.crop((0, 0, img1.width, 250))
        header_text = pytesseract.image_to_string(header_crop)
        header = extract_header_text(header_text)
        
        # --- Material Balance via positional OCR ---
        data = pytesseract.image_to_data(img1, output_type=pytesseract.Output.DATAFRAME)
        data = data[data["text"].notna() & (data["text"].str.strip() != "")]
        table = data[
            (data["top"] >= OCR_TABLE_Y_START) & (data["top"] <= OCR_TABLE_Y_END)
        ].copy()
        
        table["cx"] = table["left"] + table["width"] / 2
        
        def assign_col(cx):
            for name, lo, hi in OCR_COLUMNS:
                if lo <= cx < hi:
                    return name
            return "unknown"
        
        table["col"] = table["cx"].apply(assign_col)
        table["row_bin"] = ((table["top"] - OCR_TABLE_Y_START) // OCR_ROW_HEIGHT).astype(int)
        
        rows = []
        for row_bin, group in table.groupby("row_bin"):
            row = {}
            for col_name in [c[0] for c in OCR_COLUMNS]:
                words = group[group["col"] == col_name].sort_values("left")
                row[col_name] = " ".join(words["text"].astype(str).tolist())
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Parse numbers
        numeric_cols = [
            "storage_bom", "receipts", "input_run_blends",
            "products_mfg", "fuel_used", "deliveries", "storage_eom"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(parse_number)
        
        # Parse code, use canonical product names
        def parse_code(raw):
            s = re.sub(r"[^0-9]", "", str(raw))
            return int(s) if s else None
        df["code"] = df["code"].apply(parse_code)
        
        # Since OCR can misread codes, also assign by row position (rows are always 1-24)
        for i, code in enumerate(range(1, 25)):
            if i < len(df):
                if df.loc[df.index[i], "code"] is None or df.loc[df.index[i], "code"] != code:
                    df.loc[df.index[i], "code_by_position"] = code
                else:
                    df.loc[df.index[i], "code_by_position"] = code
        
        df["material_name"] = df.get("code_by_position", df["code"]).map(PRODUCT_CODES)
        
        return {
            "file": os.path.basename(pdf_path),
            "extraction_mode": "scanned_ocr",
            "header": header,
            "material_balance": df[
                ["code", "material_name"] + numeric_cols
            ] if not df.empty else pd.DataFrame(),
            "capacity": pd.DataFrame(),  # Would need OCR on page 4
            "page_count": len(pages),
            "receipts_raw": "",
            "deliveries_raw": "",
        }
    finally:
        for p in pages:
            os.unlink(p)


# ─── Unified extraction + validation ─────────────────────────────────────────

def validate_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check the material balance equation for each row:
    Storage_BOM + Receipts + Products_Mfg = Input/Run/Blends + Fuel_Used + Deliveries + Storage_EOM
    
    Returns the df with a 'balance_check' column (should be ~0 if correct).
    """
    if df.empty:
        return df
    
    df = df.copy()
    # Inflows: storage_bom + receipts + products_mfg
    inflows = df["storage_bom"].fillna(0) + df["receipts"].fillna(0) + df["products_mfg"].fillna(0)
    # Outflows: input_run_blends + fuel_used + deliveries + storage_eom
    outflows = (df["input_run_blends"].fillna(0) + df["fuel_used"].fillna(0) 
                + df["deliveries"].fillna(0) + df["storage_eom"].fillna(0))
    df["balance_check"] = inflows - outflows
    df["balanced"] = df["balance_check"].abs() < 1  # Allow rounding tolerance
    
    return df


def extract_r1(pdf_path: str) -> dict:
    """Auto-detect PDF type and extract accordingly."""
    digital = is_digital_pdf(pdf_path)
    
    if digital:
        result = extract_digital(pdf_path)
    else:
        result = extract_scanned(pdf_path)
    
    # Validate material balance
    if not result["material_balance"].empty:
        result["material_balance"] = validate_balance(result["material_balance"])
    
    return result


# ─── Output formatting ───────────────────────────────────────────────────────

def print_result(result: dict):
    """Pretty-print extraction results."""
    h = result["header"]
    print(f"\n{'='*70}")
    print(f"  File:      {result['file']}")
    print(f"  Mode:      {result['extraction_mode']}")
    print(f"  Operator:  {h.get('operator', '?')}")
    print(f"  Month:     {h.get('month', '?')}")
    print(f"  Reg No:    {h.get('registration_no', '?')}")
    print(f"  Refinery:  {h.get('refinery_name', h.get('refinery_location', '?'))}")
    print(f"  Pages:     {result['page_count']}")
    print(f"{'='*70}")
    
    df = result["material_balance"]
    if not df.empty:
        print("\nMaterial Balance (barrels):")
        display = df[["code", "material_name", "storage_bom", "receipts",
                       "input_run_blends", "products_mfg", "fuel_used",
                       "deliveries", "storage_eom"]].copy()
        # Format numbers with commas
        for col in display.columns:
            if col not in ("code", "material_name"):
                display[col] = display[col].apply(
                    lambda x: f"{x:,.0f}" if pd.notna(x) else "-"
                )
        print(display.to_string(index=False))
        
        # Validation summary
        if "balanced" in df.columns:
            n_ok = df["balanced"].sum()
            n_total = len(df[df[["storage_bom","receipts","products_mfg",
                                  "deliveries","storage_eom"]].notna().any(axis=1)])
            print(f"\nBalance check: {n_ok}/{n_total} rows pass")
            fails = df[~df["balanced"] & df[["storage_bom","receipts","products_mfg",
                                              "deliveries","storage_eom"]].notna().any(axis=1)]
            if not fails.empty:
                print("  Mismatches:")
                for _, row in fails.iterrows():
                    print(f"    Code {row['code']:2.0f} ({row['material_name']}): "
                          f"off by {row['balance_check']:,.0f} bbl")
    
    cap = result.get("capacity", pd.DataFrame())
    if not cap.empty:
        print("\nStorage Capacity vs End-of-Month Inventory:")
        cap = cap.copy()
        cap["utilization_pct"] = (cap["storage_eom"] / cap["capacity"] * 100).round(1)
        cap["utilization_pct"] = cap["utilization_pct"].where(cap["capacity"] > 0)
        for col in ["capacity", "storage_eom"]:
            cap[col] = cap[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
        cap["utilization_pct"] = cap["utilization_pct"].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "-"
        )
        print(cap[["code", "material_name", "capacity", "storage_eom", "utilization_pct"]]
              .to_string(index=False))


def export_to_csv(results: list[dict], output_path: str = "rrc_r1_database.csv"):
    """Combine multiple R-1 extractions into a single CSV for database import."""
    frames = []
    for r in results:
        df = r["material_balance"].copy()
        df["operator"] = r["header"].get("operator")
        df["month"] = r["header"].get("month")
        df["registration_no"] = r["header"].get("registration_no")
        df["refinery_name"] = r["header"].get("refinery_name", "")
        df["source_file"] = r["file"]
        df["extraction_mode"] = r["extraction_mode"]
        frames.append(df)
    
    combined = pd.concat(frames, ignore_index=True)
    
    # Reorder columns for readability
    front_cols = ["operator", "refinery_name", "registration_no", "month",
                  "code", "material_name"]
    data_cols = ["storage_bom", "receipts", "input_run_blends", "products_mfg",
                 "fuel_used", "deliveries", "storage_eom"]
    meta_cols = ["balance_check", "balanced", "source_file", "extraction_mode"]
    col_order = [c for c in front_cols + data_cols + meta_cols if c in combined.columns]
    combined = combined[col_order]
    
    combined.to_csv(output_path, index=False)
    print(f"\nExported {len(combined)} rows to {output_path}")
    return combined


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    target = sys.argv[1]
    
    if os.path.isdir(target):
        pdfs = sorted(Path(target).glob("*.pdf"))
        print(f"Found {len(pdfs)} PDFs in {target}")
    else:
        pdfs = [Path(target)]
    
    results = []
    for pdf in pdfs:
        try:
            result = extract_r1(str(pdf))
            results.append(result)
            print_result(result)
        except Exception as e:
            print(f"\nERROR processing {pdf}: {e}")
    
    if len(results) > 1:
        export_to_csv(results)
    elif len(results) == 1:
        export_to_csv(results, "rrc_r1_single.csv")


if __name__ == "__main__":
    main()
