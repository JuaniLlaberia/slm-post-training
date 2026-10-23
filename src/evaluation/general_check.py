GENERAL = [
    # Writing
    {"id": "w1", "category": "writing",
     "prompt": "Write a two-sentence email declining a meeting invitation.",
     "rule": "Exactly 2 sentences, polite, clearly declines."},
    {"id": "w2", "category": "writing",
     "prompt": "Write a short thank-you note to a friend who helped you move house.",
     "rule": "A thank-you note that mentions the help with moving; under about 80 words."},
    {"id": "w3", "category": "writing",
     "prompt": "Give one practical tip for sleeping better.",
     "rule": "One sensible tip, not a list of several."},
    {"id": "w4", "category": "writing",
     "prompt": "Write a one-sentence description of a rainy day.",
     "rule": "Exactly 1 sentence about a rainy day."},

    # Following a format
    {"id": "f1", "category": "format",
     "prompt": "List three fruits as a JSON array of strings.",
     "rule": "Valid JSON array with exactly 3 fruit names as strings."},
    {"id": "f2", "category": "format",
     "prompt": "Write the days of the week in a numbered list.",
     "rule": "Numbered list of the 7 days, in order."},
    {"id": "f3", "category": "format",
     "prompt": "Answer with only yes or no: Is the Sun a star?",
     "rule": "Says 'yes' and nothing more than a word or two."},
    {"id": "f4", "category": "format",
     "prompt": "Write the word 'hello' in uppercase letters.",
     "rule": "Outputs HELLO."},

    # Translation
    {"id": "t1", "category": "translation",
     "prompt": "Translate into Spanish: Good morning, how are you?",
     "rule": "Correct Spanish, e.g. 'Buenos días, ¿cómo estás?' (usted form also passes)."},
    {"id": "t2", "category": "translation",
     "prompt": "Translate into English: Me gusta leer libros por la noche.",
     "rule": "Correct meaning: 'I like to read books at night.'"},

    # Simple facts
    {"id": "k1", "category": "facts",
     "prompt": "What is the capital of Japan?",
     "rule": "Says Tokyo."},
    {"id": "k2", "category": "facts",
     "prompt": "Which planet is the largest in our solar system?",
     "rule": "Says Jupiter."},
    {"id": "k3", "category": "facts",
     "prompt": "Who wrote the play Romeo and Juliet?",
     "rule": "Says William Shakespeare."},
    {"id": "k4", "category": "facts",
     "prompt": "What is the chemical symbol for gold?",
     "rule": "Says Au."},

    # Rewriting and summarizing
    {"id": "r1", "category": "rewrite",
     "prompt": "Summarize in one sentence: The town library will close for two weeks in March "
               "while workers replace the roof and repair water damage in the reading room. "
               "During the closure, residents can return books at the community center, and "
               "online services will stay available.",
     "rule": "One sentence that mentions the temporary closure and why (repairs)."},
    {"id": "r2", "category": "rewrite",
     "prompt": "Rewrite this sentence to sound more formal: hey, can u send me the report asap?",
     "rule": "Formal rewording that still asks for the report soon."},
    {"id": "r3", "category": "rewrite",
     "prompt": "Fix the spelling in this sentence: I recieved the pakage yesturday.",
     "rule": "'I received the package yesterday.'"},

    # Simple non-math judgment
    {"id": "j1", "category": "judgment",
     "prompt": "Is this review positive or negative? 'The food was cold and the service was slow.'",
     "rule": "Says negative."},
    {"id": "j2", "category": "judgment",
     "prompt": "Which word does not belong: apple, banana, carrot, grape?",
     "rule": "Says carrot (a vegetable; the others are fruits)."},
    {"id": "j3", "category": "judgment",
     "prompt": "Name the opposite of the word 'ancient'.",
     "rule": "A valid antonym such as modern, new or recent."},
]
