import os
import pandas as pd
from dotenv import load_dotenv
from ollama import chat


# Load data
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
excel_path = "Data Sample for Altro AI.xlsx"
sheet_name = "REAL and Mocked up Data for POC"
xls = pd.ExcelFile(excel_path)
df = pd.read_excel(xls, sheet_name=sheet_name).fillna("")


def generate_llm_description(row):
    title = row["Project title"]
    prompt = f"""
You are an assistant that generates 4–5 sentence warm descriptions of Art of Living projects based on what you already know for the given project title:

Title: {title}
"""

    response = chat(model=MODEL_NAME, messages=[
        {"role": "system", "content": "You write informative and inspiring blurbs for service projects."},
        {"role": "user", "content": prompt}
    ])
    return response["message"]["content"]


descriptions = []
for _, row in df.iterrows():
    try:
        if row.get("Generated Description", "").strip():
            descriptions.append(row["Generated Description"])
            continue
        desc = generate_llm_description(row)
        descriptions.append(desc)
    except Exception as e:
        print(f"Error on row {_}: {e}")
        descriptions.append("")

df["Generated Description"] = descriptions

# Write back only this sheet (preserving other sheets if any)
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df.to_excel(writer, index=False, sheet_name=sheet_name)

print("✅ File updated: Descriptions saved in-place.")