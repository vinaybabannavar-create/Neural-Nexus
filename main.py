"""
main.py — CLI entrypoint for the Corrective RAG pipeline.

Usage
-----
    # Ask a question via terminal
    python main.py query "What is contextual chunking?"

    # Ingest documents
    python main.py ingest --source docs_sample/

    # Start the FastAPI server
    python main.py serve

    # Start the Streamlit UI
    python main.py ui
"""
import sys
import argparse
from loguru import logger


def run_query(question: str):
    """Run the RAG pipeline and print the result."""
    from app.graph.pipeline import rag_graph

    print(f"\nQUERY: {question}\n")
    print("=" * 60)

    initial_state = {
        "question": question,
        "documents": [],
        "generation": None,
        "web_search_used": False,
        "retry_count": 0,
        "relevance_score": 0.0,
        "sources": [],
    }

    result = rag_graph.invoke(initial_state)

    print(f"\nANSWER:\n{result.get('generation', 'No answer generated.')}")
    print("\n" + "=" * 60)
    print(f"Web search used   : {result.get('web_search_used', False)}")
    print(f"Relevance score   : {result.get('relevance_score', 0):.2%}")
    print(f"Generation retries: {result.get('retry_count', 0)}")
    sources = result.get("sources", [])
    if sources:
        print(f"Sources           :")
        for src in sources:
            print(f"   - {src}")
    print()


def run_serve():
    """Start the FastAPI server."""
    import uvicorn
    logger.info("Starting FastAPI server on http://localhost:8000")
    logger.info("API docs available at http://localhost:8000/docs")
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)


def run_ui():
    """Start the Streamlit UI."""
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/ui.py"])


def run_ingest(source: str):
    """Ingest documents."""
    from app.ingest import ingest
    ingest(source)


def main():
    parser = argparse.ArgumentParser(
        description="Corrective RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # query
    q_parser = subparsers.add_parser("query", help="Ask a question")
    q_parser.add_argument("question", type=str, help="Your question")

    # ingest
    i_parser = subparsers.add_parser("ingest", help="Ingest documents")
    i_parser.add_argument("--source", required=True, help="File, folder, or URL")

    # serve
    subparsers.add_parser("serve", help="Start FastAPI server")

    # ui
    subparsers.add_parser("ui", help="Start Streamlit UI")

    args = parser.parse_args()

    if args.command == "query":
        run_query(args.question)
    elif args.command == "ingest":
        run_ingest(args.source)
    elif args.command == "serve":
        run_serve()
    elif args.command == "ui":
        run_ui()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
