import itertools

DEMOGRAPHICS = {
    "age":    ["25-year-old", "45-year-old", "65-year-old"],
    "gender": ["man", "woman"],
    "race":   ["White", "Black", "Hispanic", "Asian"],
    "income": ["earning under $30k a year",
            "earning $50,000-$75,000 a year",
            "earning over $100k a year"],
}

def build_personas():
    """Generate all demographic persona combinations"""
    return list(itertools.product(
        DEMOGRAPHICS['age'],
        DEMOGRAPHICS['gender'],
        DEMOGRAPHICS['race'],
        DEMOGRAPHICS['income']
    ))

def build_prompt(age, gender, race, income):
    """Build a survey prompt for a given persona"""
    return f"""You are a {age} {race} {gender} {income}.
Answer the following survey question honestly based on your background and perspective.

Question: How much do you think AI technology will affect your job opportunities?
1 = AI will create mostly new opportunities for workers like me
2 = AI will create more opportunities than it will hurt workers like me
3 = AI will hurt more workers like me than it will create opportunities
4 = AI will mostly hurt workers like me

Respond with ONLY the number (1, 2, 3, or 4). Nothing else."""

def get_llm_response(client, prompt, model="gpt-3.5-turbo"):
    """Get LLM response for a given prompt"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=5
    )
    return response.choices[0].message.content.strip()