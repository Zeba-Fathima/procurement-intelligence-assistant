from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.rag_pipeline import load_vectorstore
from src.sql_agent import ask_sql

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


def classify_question(question):
    prompt = f"""
You are a routing assistant for a Procurement Intelligence Assistant.

Decide which route should answer the user's question.

Routes:

1. sql
Use when the question needs calculations, totals, averages, rankings, counts,
highest/lowest values, supplier statistics, category statistics, cost, savings,
lead time, defect rate, compliance count, or risk score.

2. rag
Use when the question needs explanation, summary, interpretation, details,
recommendations, qualitative insights, supplier performance, or procurement issues.

3. hybrid
Use when the question needs both exact SQL calculation and explanation from RAG.

User question:
{question}

Return only one word:
sql
rag
hybrid
"""

    response = llm.invoke(prompt)
    route = response.content.strip().lower()

    if route not in ["sql", "rag", "hybrid"]:
        return "hybrid"

    return route


def ask_rag(question):
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a Procurement Intelligence Assistant.

Use only the procurement context below to answer the question.

Question:
{question}

Context:
{context}

Your answer should include:
1. Direct answer
2. Relevant procurement insight
3. Explanation of the insight

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "retrieved_docs": docs
    }


def ask_hybrid(question):
    sql_result = ask_sql(question)
    rag_result = ask_rag(question)

    prompt = f"""
You are a Procurement Intelligence Assistant.

Combine the SQL result and RAG result into one useful business answer.

User Question:
{question}

SQL Answer:
{sql_result["answer"]}

RAG Answer:
{rag_result["answer"]}

Final Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sql_query": sql_result["sql_query"],
        "sql_result": sql_result["sql_result"],
        "retrieved_docs": rag_result["retrieved_docs"]
    }


def answer_question(question):
    route = classify_question(question)

    if route == "sql":
        sql_result = ask_sql(question)

        return {
            "route": "sql",
            "answer": sql_result["answer"],
            "sql_query": sql_result["sql_query"],
            "sql_result": sql_result["sql_result"],
            "retrieved_docs": []
        }

    if route == "rag":
        rag_result = ask_rag(question)

        return {
            "route": "rag",
            "answer": rag_result["answer"],
            "sql_query": None,
            "sql_result": None,
            "retrieved_docs": rag_result["retrieved_docs"]
        }

    hybrid_result = ask_hybrid(question)

    return {
        "route": "hybrid",
        "answer": hybrid_result["answer"],
        "sql_query": hybrid_result["sql_query"],
        "sql_result": hybrid_result["sql_result"],
        "retrieved_docs": hybrid_result["retrieved_docs"]
    }