import os
import streamlit as st

from src.data_setup import clean_and_create_database
from src.rag_pipeline import build_vectorstore
from src.qa_engine import answer_question

RAW_DATA_PATH = "data/procurement_data.csv"
CLEANED_DATA_PATH = "data/cleaned_procurement_data.csv"
VECTORSTORE_PATH = "vectorstore"

st.set_page_config(
    page_title="Procurement Intelligence Assistant",
    page_icon="📦",
    layout="centered"
)

st.title("📦 Procurement Intelligence Assistant")
st.write("Ask procurement questions using SQL Agent + RAG.")


@st.cache_resource
def initialize_project():
    if not os.path.exists(CLEANED_DATA_PATH):
        clean_and_create_database(
            RAW_DATA_PATH,
            CLEANED_DATA_PATH
        )

    if not os.path.exists(VECTORSTORE_PATH):
        build_vectorstore(CLEANED_DATA_PATH)

    return True


with st.spinner("Loading assistant..."):
    initialize_project()


question = st.text_input(
    "Ask a question",
    placeholder="Example: Which supplier has the highest total spend?"
)


if st.button("Ask"):
    if question.strip():
        with st.spinner("Thinking..."):
            result = answer_question(question)

        st.markdown("### Answer")
        st.write(result.get("answer", "No answer returned."))

        with st.expander("🔍 View Execution Details"):
            route = result.get("route", "unknown")

            st.write("**Route Used:**", route.upper())

            if route in ["sql", "hybrid"]:
                st.subheader("SQL Query")
                st.code(
                    result.get("sql_query", "SQL query not available."),
                    language="sql"
                )

                st.subheader("SQL Result")
                st.text(
                    result.get("sql_result", "SQL result not available.")
                )

            if route in ["rag", "hybrid"]:
                st.subheader("Retrieved Documents")

                retrieved_docs = result.get("retrieved_docs", [])

                if retrieved_docs:
                    for i, doc in enumerate(retrieved_docs, start=1):
                        st.markdown(f"**Document {i}**")
                        st.text(doc.page_content)
                else:
                    st.write("No retrieved documents available.")

    else:
        st.warning("Please enter a question.")