import pandas as pd

from analytics import calculate_kpis, prepare_data


def test_prepare_data_parses_dates_and_calculates_revenue():
    data = pd.DataFrame(
        [
            {
                "order_id": "O1",
                "customer_id": "C1",
                "order_date": "2025-01-15",
                "category": "Home",
                "region": "West",
                "quantity": 2,
                "unit_price": 12.5,
            }
        ]
    )

    result = prepare_data(data)

    assert result.loc[0, "revenue"] == 25.0
    assert result.loc[0, "month"] == pd.Timestamp("2025-01-01")


def test_calculate_kpis_counts_repeat_customers_and_orders():
    data = prepare_data(
        pd.DataFrame(
            [
                ["O1", "C1", "2025-01-01", "Home", "West", 1, 10],
                ["O2", "C1", "2025-01-02", "Home", "West", 2, 10],
                ["O3", "C2", "2025-01-03", "Home", "East", 1, 20],
            ],
            columns=[
                "order_id",
                "customer_id",
                "order_date",
                "category",
                "region",
                "quantity",
                "unit_price",
            ],
        )
    )

    result = calculate_kpis(data)

    assert result == {
        "revenue": 50.0,
        "orders": 3,
        "active_customers": 2,
        "repeat_rate": 0.5,
        "average_order_value": 50 / 3,
    }


def test_calculate_kpis_returns_zeroes_for_empty_data():
    data = prepare_data(
        pd.DataFrame(
            columns=[
                "order_id",
                "customer_id",
                "order_date",
                "category",
                "region",
                "quantity",
                "unit_price",
            ]
        )
    )

    assert calculate_kpis(data)["repeat_rate"] == 0.0