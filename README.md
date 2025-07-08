# ai-price-fetcher

# 📊 Product Price Sorter

This is a simple Python script that takes a list of product offers from different websites and sorts them based on:

1. **Price confidence** (High > Medium > Low > Unknown)
2. **Estimated price** (Lowest first)
3. **Match score** (Highest first)

This is useful when aggregating product prices from multiple sources and you want to prioritize accurate and affordable results.

---

## 🔧 How It Works

Each product entry contains:
- A product link
- A product name
- A match score (how closely the result matches the query)
- A `price` object with:
  - `estimatedPrice`: string that may include currency symbols
  - `currency`: e.g., `GBP`
  - `confidence`: one of `high`, `medium`, `low`, `unknown`, or `null`

The script:
- Strips currency symbols and parses the price
- Assigns a rank to the confidence level
- Sorts using `(confidence rank, price, -match_score)`

---

## 🧪 Example Input

```json
[
  {
    "link": "...",
    "productName": "...",
    "matchScore": 66.67,
    "price": {
      "estimatedPrice": "£599.00",
      "currency": "GBP",
      "confidence": "high"
    }
  }
]