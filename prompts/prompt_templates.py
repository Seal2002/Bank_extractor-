"""
Prompt Templates - Pure Structure-Based Extraction (NO KEYWORD ASSUMPTIONS)
"""

SINGLE_PAGE_TRANSACTION_PROMPT = """You are extracting transactions from page {page_num} of {total_pages}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 STEP 1: IDENTIFY TABLE STRUCTURE (CRITICAL!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Banks use 3 COMMON FORMATS. Identify which one you're seeing:

📊 FORMAT 1: SEPARATE DEBIT/CREDIT COLUMNS (e.g., HDFC, ICICI, SBI)
Example: "Date | Description | Ref | Withdrawal | Deposit | Balance"
- Has separate columns for debits and credits
- Look for headers: "Withdrawal"/"Debit"/"Dr"/"Paid Out" (DEBIT COLUMN)
- Look for headers: "Deposit"/"Credit"/"Cr"/"Paid In" (CREDIT COLUMN)

📊 FORMAT 2: AMOUNT + TYPE COLUMNS (e.g., PNB, Canara)
Example: "Date | Description | Amount | Type | Balance"
- Has ONE amount column + separate "Type" column
- Type column contains: "CR"/"DR" or "Credit"/"Debit"
- Read amount from Amount column, determine debit/credit from Type column

📊 FORMAT 3: AMOUNT WITH SUFFIX (e.g., Union Bank, BOB)
Example: "Date | Description | Amount | Balance"
Where Amount values look like: "300.00 (Dr)" or "500.00 (Cr)"
- Amount column contains the value AND the type indicator as suffix
- Look for "(Dr)", "(CR)", "(Debit)", "(Credit)" in the amount value itself
- Extract number, use suffix to determine if debit or credit

ALSO WATCH FOR:
- "Value Date" or "Value Dt" column → IGNORE IT (it's not an amount!)
- "Cheque No" or "Ref No" column → Extract as transaction_id if present
- Multi-line descriptions → Merge them into single description field

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔢 STEP 2: READ EACH TRANSACTION ROW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each transaction row, extract data based on the format you identified:

🔹 IF FORMAT 1 (Separate Debit/Credit Columns):
   - Read date from Date column
   - Read description from Description/Narration/Particulars column
   - Read reference from Ref/Cheque No column (if exists)
   - Read amount from DEBIT column → Put in "debit" field, set "credit": null
   - Read amount from CREDIT column → Put in "credit" field, set "debit": null
   - Read balance from Balance column

🔹 IF FORMAT 2 (Amount + Type Columns):
   - Read date from Date column
   - Read description from Description/Remarks column
   - Read reference from Ref/Instrument ID column (if exists)
   - Read amount from Amount column
   - Check Type column:
     * If Type = "DR"/"Debit"/"Dr" → Put amount in "debit" field, set "credit": null
     * If Type = "CR"/"Credit"/"Cr" → Put amount in "credit" field, set "debit": null
   - Read balance from Balance column

🔹 IF FORMAT 3 (Amount with Suffix):
   - Read date from Date column
   - Read description from Description/Remarks column
   - Read reference from Ref column (if exists)
   - Look at Amount column value:
     * If contains "(Dr)" or "(DR)" or "(Debit)" → Extract number, put in "debit" field, set "credit": null
     * If contains "(Cr)" or "(CR)" or "(Credit)" → Extract number, put in "credit" field, set "debit": null
   - Read balance from Balance column

🚨 CRITICAL RULES (READ CAREFULLY):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ONE transaction = ONE row in the table (don't create duplicates)
2. For each transaction, ONLY ONE of debit/credit should have a value (the other should be null)
3. If you see "Value Date" column, SKIP it when reading amounts
4. Extract transaction_id from any reference/cheque/instrument column if present
5. Merge multi-line descriptions into single field
6. Skip header rows, footer rows ("Total", "Continued", etc.)
7. Remove suffixes like "(Dr)", "(Cr)" from the amount value - just extract the number
8. Don't make assumptions based on description keywords - rely on column structure/Type indicator

⚠️ EXAMPLES FOR EACH FORMAT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Example 1 - FORMAT 1 (HDFC):
Row: "01/10/18 | CHQ PAID | 300 | 01/10/18 | 25000.00 | | 4977.83"
      Date    | Desc      | Ref | Value Dt| Withdrawal| Deposit | Balance

Extract as:
{{
  "date": "2018-10-01",
  "description": "CHQ PAID",
  "transaction_id": "300",
  "debit": 25000.00,    ← From Withdrawal column
  "credit": null,        ← Deposit column is empty
  "balance": 4977.83
}}

Example 2 - FORMAT 2 (PNB):
Row: "09/06/2025 | | 410.30 | CR | 21681.31 | NEFT_IN:..."
      Date      |ID| Amount | Type| Balance | Remarks

Extract as:
{{
  "date": "2025-06-09",
  "description": "NEFT_IN:...",
  "transaction_id": null,
  "debit": null,
  "credit": 410.30,      ← Amount column, Type=CR
  "balance": 21681.31
}}

Example 3 - FORMAT 3 (Union Bank):
Row: "17/02/2022 | UPIAR/... | 300.00 (Dr) | 85.57"
      Date      | Remarks  | Amount      | Balance

Extract as:
{{
  "date": "2022-02-17",
  "description": "UPIAR/...",
  "transaction_id": null,
  "debit": 300.00,       ← Amount has (Dr) suffix
  "credit": null,
  "balance": 85.57
}}

Another Example 3:
Row: "14/01/2022 | NEFT:... | 10000.00 (Cr) | 10966.07"
      Date      | Remarks | Amount        | Balance

Extract as:
{{
  "date": "2022-01-14",
  "description": "NEFT:...",
  "transaction_id": null,
  "debit": null,
  "credit": 10000.00,    ← Amount has (Cr) suffix
  "balance": 10966.07
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📤 STEP 3: OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY a JSON array (no markdown, no explanation):

[
  {{
    "date": "YYYY-MM-DD",
    "description": "Clean text (merge multi-line if needed, remove 'S DEBIT' suffix)",
    "transaction_id": "Reference number or null",
    "debit": 100.50,
    "credit": null,
    "balance": 5000.00
  }},
  {{
    "date": "YYYY-MM-DD",
    "description": "Another transaction",
    "transaction_id": "REF123",
    "debit": null,
    "credit": 200.00,
    "balance": 5200.00
  }}
]

IMPORTANT:
- Dates must be YYYY-MM-DD format & consistent & squential
- debit and credit: one should be null, other should be number (or both can have values if table shows both)
- Remove currency symbols (₹, Rs.) and commas from numbers
- Clean up descriptions (merge multi-line, remove common suffixes like "S DEBIT", "PO S DEBIT")
- Extract transaction_id from reference column if present
- Skip footer text like "Continued on next page", "Total", etc.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ FINAL VERIFICATION BEFORE RETURNING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before returning your JSON:
1. Did I extract ALL transaction rows from page {page_num}?
2. Did I read amounts from correct column positions (not based on keywords)?
3. Did I skip the Value Date column if present?
4. Did I avoid creating duplicate rows for the same transaction?
5. Are all dates in YYYY-MM-DD format?
6. Did I clean up descriptions (merged multi-line, removed suffixes)?

Extract page {page_num} now:"""


TRANSACTION_EXTRACTION_PROMPT = """Extract ALL transactions from this bank statement PDF.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 YOUR MISSION: Identify Format → Extract Data (NO ASSUMPTIONS!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Banks use 3 COMMON TABLE FORMATS:

📊 FORMAT 1: SEPARATE DEBIT/CREDIT COLUMNS (e.g., HDFC, ICICI)
"Date | Description | Ref | Withdrawal | Deposit | Balance"
- Separate columns for debits and credits
- Read from Withdrawal/Debit/Dr column → "debit" field
- Read from Deposit/Credit/Cr column → "credit" field

📊 FORMAT 2: AMOUNT + TYPE COLUMNS (e.g., PNB, Canara)
"Date | Description | Amount | Type | Balance"
- Single Amount column + separate Type column (CR/DR)
- Read amount, check Type column to determine debit or credit

📊 FORMAT 3: AMOUNT WITH SUFFIX (e.g., Union Bank, BOB)
"Date | Description | Amount | Balance"
Values like: "300.00 (Dr)" or "500.00 (Cr)"
- Amount value contains (Dr) or (Cr) suffix
- Extract number, use suffix to determine debit or credit

CRITICAL RULES:
✓ Identify which format the statement uses
✓ For FORMAT 1: Read by column position (Withdrawal→debit, Deposit→credit)
✓ For FORMAT 2: Read Amount column + check Type column (CR/DR)
✓ For FORMAT 3: Extract number from Amount, use (Dr)/(Cr) suffix
✓ Ignore "Value Date" or "Value Dt" columns (not amounts!)
✓ ONE transaction = ONE row (don't duplicate)
✓ Only ONE of debit/credit should have value per transaction (other is null)

OUTPUT FORMAT:
Return ONLY JSON array (no markdown):

[
  {{
    "date": "YYYY-MM-DD",
    "description": "Clean text (remove 'S DEBIT' suffix if present)",
    "transaction_id": "Reference number or null",
    "debit": 100.50 or null,
    "credit": null or 50.25,
    "balance": 1000.00
  }}
]

IMPORTANT:
- Convert dates to YYYY-MM-DD format
- debit and credit are usually mutually exclusive (one should be null)
- Extract transaction_id if visible in reference column
- Remove currency symbols (₹, Rs.) and commas
- Merge multi-line descriptions into single field
- Skip footer text and summary rows

Extract ALL transactions now:"""


METADATA_EXTRACTION_PROMPT = """Extract account metadata from this bank statement.

Look for:
- Account holder name
- Account number
- Bank name / branch
- Statement period (from date, to date)
- Opening balance (at start of statement)
- Closing balance (at end of statement)

Return ONLY JSON object (no markdown):

{{
  "account_holder_name": "Full Name",
  "account_number": "XXXX1234",
  "bank_name": "Bank Name",
  "branch": "Branch Name",
  "statement_period_from": "YYYY-MM-DD",
  "statement_period_to": "YYYY-MM-DD",
  "opening_balance": 5000.00,
  "closing_balance": 6000.00,
  "currency": "INR"
}}

Extract metadata now:"""