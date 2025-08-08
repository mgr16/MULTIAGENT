import argparse
import asyncio
from app.main import run_query

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--q", "--query", dest="query", required=True, help="Pregunta/consulta de usuario")
    p.add_argument("--image", dest="image_url", default=None, help="URL/base64 de imagen opcional")
    args = p.parse_args()
    ans, usage = asyncio.run(run_query(args.query, image_url=args.image_url))
    print("=== ANSWER ===")
    print(ans)
    print("\n=== USAGE ===")
    print("\n".join(usage))

if __name__ == "__main__":
    main()
