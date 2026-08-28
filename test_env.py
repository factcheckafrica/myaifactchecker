from dotenv import load_dotenv
import os

load_dotenv()

keys = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
    "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
    "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
}

print("\n" + "="*60)
print("ENVIRONMENT VARIABLE TEST")
print("="*60)

for name, value in keys.items():
    if value:
        masked = f"{value[:10]}...{value[-4:]}"
        print(f"[OK] {name}: {masked} ({len(value)} chars)")
    else:
        print(f"[MISSING] {name}: NOT FOUND")

print("="*60 + "\n")
