#!/usr/bin/env bash
# Minimal end-to-end check against a running jev-bridge.
set -euo pipefail
BASE="${1:-http://127.0.0.1:8900}"

echo "--- health ---"
curl -s "$BASE/health" | python3 -m json.tool

echo "--- systemone (choice + noul + score) ---"
curl -s "$BASE/v1/systemone" -H 'Content-Type: application/json' -d '{
  "model": "jev-latest",
  "state": "My card was charged twice for order A-104 and I need this fixed today.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing":  "Charges, invoices, payment problems",
        "shipping": "Delivery status, delays, lost packages",
        "returns":  "Exchanges, refunds, damaged items"
      }
    },
    "is_urgent": {"type": "noul", "instructions": "Does this message express urgency?"},
    "severity": {
      "type": "score",
      "instructions": "How severe is the issue?",
      "criteria": ["Cosmetic", "Workaround exists", "Blocking, no workaround"]
    }
  }
}' | python3 -m json.tool
