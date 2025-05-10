# %%
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa

tokenizer = LlamaTokenizer.from_pretrained("lzw1008/Emollama-7b")
model = LlamaForCausalLM.from_pretrained("lzw1008/Emollama-7b", device_map="auto")


# %%
def get_answer(prompt: str, max_length=500):
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

    output = model.generate(
        **input_ids,
        # https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id
        pad_token_id=tokenizer.eos_token_id,
        max_length=max_length,
    )

    return tokenizer.batch_decode(output, skip_special_tokens=True)


# %%
get_answer("""why the sky is blue?""", max_length=300)

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
Text: Попробуй добавить 'kill' в вызов функции `naked_processing()`.
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
