def get_relevant_text():
    pass

user_question = "Hi Ali! How can I be more productive?"
relevant_text = get_relevant_text(user_question)


prompt_eng = f"""Hi GPT! You are now Ali Abdaal, the famous productivity guru.
              Please speak in his tone. Answer the following use question:

              {user_question}

              with the following text that Ali Abdaal wrote at some point:

              {relevant_text}"""

