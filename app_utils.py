import pandas as pd
from openai import OpenAI
import os
import tiktoken
from scipy import spatial  # for calculating vector similarities for search

EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"
df = pd.read_csv("ali_abdaals_articles.csv")
# client = client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ))



# prompt_eng = f"""Hi GPT! You are now Ali Abdaal, the famous productivity guru.
#               Please speak in his tone. Answer the following user question:

#               {user_question}

#               with the following text that Ali Abdaal wrote at some point:

#               {relevant_text}"""


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 200
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = OpenAI.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below articles written by Ali abdaal to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nAli Abdaal article sections:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


# def ask(
#     query: str,
#     df: pd.DataFrame = df,
#     model: str = GPT_MODEL,
#     token_budget: int = 2000,
#     print_message: bool = False,
# ) -> str:
#     """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
#     message = query_message(query, df, model=model, token_budget=token_budget)
#     if print_message:
#         print(message)
#     messages = [
#         {"role": "system", "content": "You answer questions in Ali abdaal's tone."},
#         {"role": "user", "content": message},
#     ]
#     response = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=0
#     )
#     response_message = response.choices[0].message.content
#     return response_message

