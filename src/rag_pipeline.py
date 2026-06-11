import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTOR_DB_DIR = "vectorstore"


def create_documents(csv_path):
    df = pd.read_csv(csv_path)

    documents = []

    for _, row in df.iterrows():
        text = f"""
Purchase Order ID: {row['po_id']}
Supplier: {row['supplier']}
Order Date: {row['order_date']}
Delivery Date: {row['delivery_date']}
Item Category: {row['item_category']}
Order Status: {row['order_status']}
Compliance: {row['compliance']}
Quantity: {row['quantity']}
Unit Price: {row['unit_price']}
Negotiated Price: {row['negotiated_price']}
Defective Units: {row['defective_units']}
Lead Time Days: {row['lead_time_days']}
Total Cost: {row['total_cost']}
Negotiated Total Cost: {row['negotiated_total_cost']}
Cost Savings: {row['cost_savings']}
Defect Rate: {row['defect_rate']}
Status Score: {row['status_score']}
Status Risk: {row['status_risk']}
Compliance Score: {row['compliance_score']}
Compliance Risk: {row['compliance_risk']}
Risk Score: {row['risk_score']}
Risk Level: {row['risk_level']}
"""

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "po_id": row["po_id"],
                    "supplier": row["supplier"],
                    "category": row["item_category"],
                    "compliance": row["compliance"],
                    "risk_level": row["risk_level"],
                    "source": "procurement_dataset"
                }
            )
        )

    return documents


def build_vectorstore(csv_path):
    documents = create_documents(csv_path)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    vectorstore.persist()

    print("Vector database created successfully.")


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )