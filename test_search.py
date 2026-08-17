from src.vector_db.collections import search

if __name__ == "__main__":
    hits = search("Who should be screened for hepatitis C?", k=5, corpus="uspstf")
    print(f"Found {len(hits)} results:\n")
    for i, hit in enumerate(hits, 1):
        print(f"{i}. Score: {hit['score']:.4f} | Source: {hit.get('source', 'N/A')}")
        print(f"   Heading: {hit.get('heading', 'N/A')}")
        print(f"   Text: {hit['text'][:150]}...\n")
