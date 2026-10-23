from transformers import PreTrainedTokenizerBase

EXAMPLES = [
    {
        "question": "Five shirts are bought. Of the 5 shirts, there are 3 shirts that cost $15 each. The remaining shirts are $20 each. What is the cost, in dollars, of all 5 shirts?",
        "answer": "The first 3 shirts cost 15*3=45 dollars in total.\nThere are 5-3=2 shirts remaining.\nThe remaining 2 shirts cost 20*2=40 dollars in total.\nThe five shirts cost 45+40=85 dollars.\n#### 85",
    },
    {
        "question": "Big boxes contain 7 dolls each. Small boxes contain 4 dolls each. There are 5 big boxes and 9 small boxes. How many dolls are there in total?",
        "answer": "There are 7*5=35 dolls in the big boxes.\nThere are 4*9=36 dolls in the small boxes.\nThere are 35+36=71 dolls altogether.\n#### 71",
    },
    {
        "question": "Jenny wants to heat a dish for dinner. It needs to be at 100 degrees before it is ready to eat. It is 20 degrees when she places it in the oven, and it heats up 5 degrees every minute. How many minutes will it take to be ready?",
        "answer": "To determine this, we must first figure out the required change in temperature. To do this, we perform 100-20= 80 degrees of required temperature change.\nNext, we divide this amount by the rate of change per minute, performing 80/5= 16 minutes of cooking time.\n#### 16",
    },
    {
        "question": "Omi is twice as old as Kimiko. Arlette is 3/4 times as old as Kimiko. If Kimiko is 28 years old, calculate the average age of the three?",
        "answer": "If Kimiko is 28, Omi is 2 * 28 years = 56 years old.\nArlette is 3/4 * 28 years = 21 years old.\nIf you add up their ages, it sums to 21 years + 56 years + 28 years = 105 years\nThe average age for the three is 105 years / 3 people = 35 years/person\n#### 35",
    },
    {
        "question": "Christian is twice as old as Brian. In eight more years, Brian will be 40 years old. How old will Christian be in eight years?",
        "answer": "If in eight years Brian will be 40 years old, he is currently 40-8 = 32 years old.\nSince Christian is twice as old as Brian, he is 2*32 = 64 years old.\nIn 8 years, Christian will be 64+8 = 72 years old.\n#### 72",
    },
    {
        "question": "Jenny got a 95 on her paper. Jason's grade was 25 points less than Jenny's, and Bob's grade was equal to half of Jason's grade. What was Bob's grade?",
        "answer": "First find Jason's grade by subtracting 25 from Jenny's grade: 95 - 25 = 70\nThen divide Jason's grade by 2 to find Bob's grade: 70 / 2 = 35\n#### 35",
    },
    {
        "question": "Phoebe eats 1 serving and gives her dog 1 serving of peanut butter for a bedtime snack. Each jar of peanut butter has 15 servings. How many jars will she need to make sure she and her dog have enough to last for 30 days?",
        "answer": "She and her dog both have a serving before bed so they eat 1+1 = 2 servings a day\nShe wants to have enough for 30 days and they eat 2 servings a day so they need 30*2 = 60 servings\nEach jar has 15 servings and she needs 60 servings so she will need 60/15 = 4 jars of peanut butter\n#### 4",
    },
    {
        "question": "Theo, Mason, and Roxy are siblings. Theo drinks 8 cups of water every day. Mason drinks 7 cups of water. Roxy drinks 9 cups of water every day. In one week, how many cups of water do the siblings drink together?",
        "answer": "Everyday, the three siblings drink 8 + 7 + 9 = 24 cups of water.\nIn one week, they drink a total of 24 x 7 = 168 cups of water.\n#### 168",
    },
]

STOP = ["\nQuestion:"]

def build_prompt(question: str, examples: list[dict] | None = None) -> str:
    """
    Transforms dictionary into string prompt. All examples go to the same prompt so it works for few-shot example and zero-shot.

    Args:
        question (str): Current prompt to transform.
        examples (list[dict] | None): List of dataset examples to use for few-shot prompting.
    Returns:
        str: Formatted prompt
    """
    blocks = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in (examples or [])]
    blocks.append(f"Question: {question}\nAnswer:")

    return "\n\n".join(blocks)

def build_instruct_prompt(question: str, tokenizer: PreTrainedTokenizerBase) -> str:
    """
    Formats questions/prompts into chat template format.

    Args:
        question (str): Question/prompt to format.
        tokenizer (PreTrainedTokenizerBase): Tokenizer of the chat model, used for its chat template.
    Returns:
        str: Formatted prompt.
    """
    messages = [{"role": "user",
                 "content": f"{question}\nSolve step by step. End your answer with a final line '#### <number>'."}]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
