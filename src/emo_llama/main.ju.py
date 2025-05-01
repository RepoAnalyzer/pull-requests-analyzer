# %%
print("Hello world!")

# %%
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa

tokenizer = LlamaTokenizer.from_pretrained("lzw1008/Emollama-7b")
model = LlamaForCausalLM.from_pretrained("lzw1008/Emollama-7b", device_map="auto")

# %%
input_ids = tokenizer("Why the sky is blue?", return_tensors="pt").to("cuda")
input_ids

# %%
output = model.generate(**input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))


# %%
def get_answer(prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids

    output = model.generate(**input_ids)

    return tokenizer.decode(output[0], skip_special_tokens=True)


# %% [markdown]
# ## Prompt Engineering

# %% [markdown]
# Example message in GitHub.

# %%
text = """
Text: You really didn't tried it before pushing to branch?
"""

# %% [markdown]
# ### Emotion Intensity

# %%
task = """
Task: Assign a numerical value between 0 (least E) and 1 (most E) to represent
the intensity of emotion E expressed in the text.
"""

print(get_answer(task + text))

# %% [markdown]
# ### Sentiment strength

# %%
task = """
Task: Evaluate the valence intensity of the writer's mental state based on the text,
assigning it a real-valued score from -1 (most negative) to 1 (most positive).
"""

print(get_answer(task + text))

# %% [markdown]
# ## Промпт Инжиниринг

# %% [markdown]
# Примеры сообщений на площадке GitHub.

# %%
text_en = """
Text: You really didn't tried it before pushing to branch?
"""

text_ru = """
Text: Попробуй добавить 'kill' в вызов функции `naked_processing()`.
"""

# %% [markdown]
# ### Emotion Intensity

# %%
task = """
Task: Assign a numerical value between 0 (least E) and 1 (most E) to represent
the intensity of emotion E expressed in the text.
"""

print(get_answer(task + text_en))
print(get_answer(task + text_ru))

# %% [markdown]
# ### Sentiment strength

# %%
task = """
Task: Evaluate the valence intensity of the writer's mental state based on the text,
assigning it a real-valued score from -1 (most negative) to 1 (most positive).
"""

print(get_answer(task + text_en))
print(get_answer(task + text_ru))
