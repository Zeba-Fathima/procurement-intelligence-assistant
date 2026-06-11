import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq

load_dotenv()

DB_PATH = "sqlite:///data/procurement.db"

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


def generate_sql_query(question):
    engine = create_engine(DB_PATH)

    db = SQLDatabase(
        engine,
        include_tables=["procurement_orders"]
    )

    schema = db.get_table_info()

    prompt = f"""
You are an expert SQLite query generator.

Create one valid SQLite SELECT query for the user's question.

Use only this table:
procurement_orders

Table schema:
{schema}

Rules:
- Return only the SQL query.
- Do not explain.
- Do not use markdown.
- Use SQLite syntax.
- Only use SELECT queries.
- For year filtering, use strftime('%Y', order_date).

User question:
{question}
"""

    response = llm.invoke(prompt)

    sql_query = response.content.strip()
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

    if not sql_query.endswith(";"):
        sql_query += ";"

    return sql_query


def execute_sql_query(sql_query):
    if not sql_query.lower().strip().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    engine = create_engine(DB_PATH)
    result_df = pd.read_sql_query(sql_query, engine)

    return result_df


def ask_sql(question):
    sql_query = generate_sql_query(question)
    result_df = execute_sql_query(sql_query)

    sql_result_text = result_df.to_string(index=False)

    prompt = f"""
You are a Procurement Intelligence Assistant.

Answer the user's question in a clear business-friendly sentence.

User Question:
{question}

SQL Query Used:
{sql_query}

SQL Result:
{sql_result_text}

Final Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sql_query": sql_query,
        "sql_result": sql_result_text
    }