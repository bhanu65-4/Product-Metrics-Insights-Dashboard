from __future__ import annotations

import random

import pandas as pd


REQUIRED_COLUMNS = {
    "order_id",
    "customer_id",
    "order_date",
    "category",
    "region",
    "quantity",
    "unit_price",
}


def build_sample_data(seed: int = 42) -> pd.DataFrame:
    """Create a repeatable two-year retail dataset for the first dashboard run."""
    rng = random.Random(seed)
    customer_ids = [f"C{number:04d}" for number in range(1, 651)]
    categories = ["Accessories", "Home", "Outdoor", "Technology", "Wellness"]
    regions = ["Central", "East", "South", "West"]
    category_prices = {
        "Accessories": (18, 140),
        "Home": (25, 260),
        "Outdoor": (30, 320),
        "Technology": (70, 900),
        "Wellness": (12, 180),
    }
    rows = []
    start = pd.Timestamp("2024-01-01")
    day_span = (pd.Timestamp("2025-12-31") - start).days

    for order_number in range(1, 8001):
        category = rng.choice(categories)
        customer_id = rng.choice(customer_ids)
        quantity = rng.choices([1, 2, 3, 4], weights=[58, 25, 12, 5])[0]
        rows.append(
            {
                "order_id": f"O{order_number:05d}",
                "customer_id": customer_id,
                "order_date": start + pd.Timedelta(days=rng.randint(0, day_span)),
                "category": category,
                "region": rng.choice(regions),
                "quantity": quantity,
                "unit_price": round(rng.uniform(*category_prices[category]), 2),
            }
        )

    return prepare_data(pd.DataFrame(rows))


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate a transaction CSV and add its derived revenue field."""
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    cleaned = data.copy()
    for column in REQUIRED_COLUMNS:
        cleaned[column] = cleaned[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    cleaned["order_date"] = pd.to_datetime(cleaned["order_date"], errors="coerce")
    cleaned["quantity"] = pd.to_numeric(cleaned["quantity"], errors="coerce")
    cleaned["unit_price"] = pd.to_numeric(cleaned["unit_price"], errors="coerce")
    cleaned = cleaned.dropna(
        subset=["order_id", "customer_id", "order_date", "quantity", "unit_price"]
    )

    for column in ["order_id", "customer_id", "category", "region"]:
        cleaned = cleaned[
            cleaned[column].notna() & cleaned[column].astype(str).str.strip().ne("")
        ]

    cleaned = cleaned[(cleaned["quantity"] > 0) & (cleaned["unit_price"] >= 0)]
    cleaned["revenue"] = cleaned["quantity"] * cleaned["unit_price"]
    cleaned["month"] = cleaned["order_date"].dt.to_period("M").dt.to_timestamp()
    return cleaned


def calculate_kpis(data: pd.DataFrame) -> dict[str, float | int]:
    """Calculate headline metrics using the currently filtered transactions."""
    required_fields = {"customer_id", "order_id"}
    if data.empty or not required_fields.issubset(data.columns):
        return {
            "revenue": 0.0,
            "orders": 0,
            "active_customers": 0,
            "repeat_rate": 0.0,
            "average_order_value": 0.0,
        }

    orders_per_customer = data.groupby("customer_id")["order_id"].nunique()
    active_customers = int(orders_per_customer.size)
    repeat_customers = int((orders_per_customer > 1).sum())
    order_count = int(data["order_id"].nunique())
    revenue = float(data["revenue"].sum()) if "revenue" in data.columns else 0.0

    return {
        "revenue": revenue,
        "orders": order_count,
        "active_customers": active_customers,
        "repeat_rate": repeat_customers / active_customers if active_customers else 0.0,
        "average_order_value": revenue / order_count if order_count else 0.0,
    }