import pandas as pd
from sqlalchemy import create_engine

DB_PATH = "sqlite:///data/procurement.db"


def clean_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def clean_and_create_database(
    input_path="data/procurement_data.csv",
    output_path="data/cleaned_procurement_data.csv"
):
    df = pd.read_csv(input_path)

    df = clean_column_names(df)

    text_cols = [
        "po_id",
        "supplier",
        "item_category",
        "order_status",
        "compliance"
    ]

    for col in text_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")

    numeric_cols = [
        "quantity",
        "unit_price",
        "negotiated_price",
        "defective_units"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["quantity"] = df["quantity"].fillna(df["quantity"].median())
    df["unit_price"] = df["unit_price"].fillna(df["unit_price"].median())
    df["negotiated_price"] = df["negotiated_price"].fillna(df["negotiated_price"].median())
    df["defective_units"] = df["defective_units"].fillna(0)

    df["order_date"] = df["order_date"].fillna(df["order_date"].mode()[0])
    df["delivery_date"] = df["delivery_date"].fillna(df["delivery_date"].mode()[0])

    df = df.drop_duplicates()

    df["lead_time_days"] = (
        df["delivery_date"] - df["order_date"]
    ).dt.days

    df["total_cost"] = df["quantity"] * df["unit_price"]

    df["negotiated_total_cost"] = (
        df["quantity"] * df["negotiated_price"]
    )

    df["cost_savings"] = (
        df["total_cost"] - df["negotiated_total_cost"]
    )

    df["defect_rate"] = (
        df["defective_units"] / df["quantity"]
    ) * 100

    status_score_map = {
        "delivered": 1.0,
        "partially delivered": 0.5,
        "pending": 0.3,
        "cancelled": 0.0
    }

    df["status_score"] = (
        df["order_status"]
        .map(status_score_map)
        .fillna(0.5)
    )

    df["status_risk"] = (
        1 - df["status_score"]
    ) * 100

    df["compliance_score"] = (
        df["compliance"] == "yes"
    ).astype(int)

    df["compliance_risk"] = (
        1 - df["compliance_score"]
    ) * 100

    df["risk_score"] = (
        0.30 * df["defect_rate"]
        + 0.25 * df["lead_time_days"]
        + 0.25 * df["status_risk"]
        + 0.20 * df["compliance_risk"]
    )

    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-1, 10, 25, 1000],
        labels=["Low", "Medium", "High"]
    )

    df.to_csv(output_path, index=False)

    engine = create_engine(DB_PATH)

    df.to_sql(
        "procurement_orders",
        engine,
        if_exists="replace",
        index=False
    )

    return df