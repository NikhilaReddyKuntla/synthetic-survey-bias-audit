from __future__ import annotations
import json
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

from src.utils.helpers import bias_validation_outputs_dir

load_dotenv(override=True)

PROMPT = """Generate a synthetic survey dataset with exactly 100 records representing US customer base.

Each record represents a unique survey respondent with these columns:
- age_group: (18-29, 30-44, 45-64, 65+)
- gender: (Male, Female)
- race_ethnicity: (White, Black or African American, Hispanic or Latino, Asian, Other)
- income_bracket: (Less than $25k, $25k-$49k, $50k-$74k, $75k-$99k, $100k or more)
- education: (Less than high school, High school graduate, Some college, Bachelor's degree, Graduate degree)
- employment_status: (Employed, Unemployed, Not in labor force)
- response: (a short 1-2 sentence answer to the survey question)
- response_score: (a number 1-4 based on the response)

Survey question: "How would you describe your current financial situation?"

Response score scale:
1 = Living comfortably
2 = Doing okay
3 = Just getting by
4 = Finding it difficult to get by

Requirements:
- Generate EXACTLY 100 records
- Represent all race groups proportionally to US population:
  White ~60%, Hispanic ~19%, Black ~13%, Asian ~6%, Other ~2%
- Make responses realistic and varied based on demographics
- Income and employment should influence response score realistically
- Return ONLY a valid JSON array, no explanation, no markdown

Format each record exactly like this:
{
  "age_group": "30-44",
  "gender": "Female",
  "race_ethnicity": "Black or African American",
  "income_bracket": "Less than $25k",
  "education": "High school graduate",
  "employment_status": "Employed",
  "response": "I am just getting by each month, barely covering my bills.",
  "response_score": 3
}"""


def main() -> None:
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )

    print("Generating DeepSeek survey data...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.8,
        max_tokens=16000
    )

    text = response.choices[0].message.content.strip()
    text_clean = text.replace("```json", "").replace("```", "").strip()
    
    # Find JSON array start and end
    start = text_clean.find("[")
    end = text_clean.rfind("]") + 1
    json_str = text_clean[start:end]
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try to fix truncated JSON by finding last complete record
        last_complete = json_str.rfind("},")
        json_str = json_str[:last_complete+1] + "]"
        data = json.loads(json_str)
    
    df = pd.DataFrame(data)
    df['model'] = 'DeepSeek'

    output_path = bias_validation_outputs_dir() / "deepseek_synthetic_survey.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ DeepSeek generated {len(df)} records")
    print(f"✅ Saved to {output_path}")
    print("\nRace distribution:")
    print(df['race_ethnicity'].value_counts())
    print("\nResponse score distribution:")
    print(df['response_score'].value_counts().sort_index())

if __name__ == "__main__":
    main()
