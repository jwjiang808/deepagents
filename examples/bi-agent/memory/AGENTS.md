# BI Agent Memory

This file is the agent's persistent long-term memory. It is loaded at startup
and injected into the system prompt. Update it using the `edit_file` tool
when you learn new business rules, field aliases, or correction patterns.

## Schema overview

- **sales**: id, date, region, product, amount, quantity
- **products**: id, name, category, price

## Business rules and domain knowledge

- The `date` column in `sales` uses ISO format: `YYYY-MM-DD`.
- Monetary amounts are stored in USD with two decimal places.
- Valid regions: `North`, `South`, `East`, `West`.
- Q1 covers January–March, Q2 covers April–June,
  Q3 covers July–September, Q4 covers October–December.
- To compute revenue per unit: `sales.amount / sales.quantity`.
- To join products: `JOIN products ON sales.product = products.name`.

## Learned corrections

*(Append corrections here when the user provides feedback.  Each entry
should state what was wrong and what the correct behaviour is.)*

