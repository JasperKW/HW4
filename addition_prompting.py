# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
import together
from time import sleep
import re
from sklearn.metrics import mean_absolute_error

# %%
def dprint(s, debug):
    if debug:
        print(s)

# %%
# TODO: find your API key here
# https://api.together.xyz/settings/api-keys
YOUR_API_KEY = '35aa8e364eb9f6f2b4c6e9eb61a7f124be16db13e4d5c7335d9cde39620fe0ab'
together.api_key = YOUR_API_KEY

def call_together_api(prompt, student_configs, post_processing, model='meta-llama/Llama-2-7b-chat-hf', debug=False):
    output = together.Complete.create(
    prompt = prompt,
    model = model, 
    **student_configs
    )
    dprint('*****prompt*****', debug)
    dprint(prompt, debug)
    dprint('*****result*****', debug)
    res = output['output']['choices'][0]['text']
    dprint(res, debug)
    dprint('*****output*****', debug)
    numbers_only = post_processing(res)
    dprint(numbers_only, debug)
    dprint('=========', debug)
    return numbers_only


# %% [markdown]
# ###  Part 1. Zero Shot Addition

# %%
def get_addition_pairs(lower_bound, upper_bound, rng):
    int_a = int(np.ceil(rng.uniform(lower_bound, upper_bound)))
    int_b = int(np.ceil(rng.uniform(lower_bound, upper_bound)))
    return int_a, int_b

def test_range(added_prompt, prompt_configs, rng, n_sample=30, 
               lower_bound=1, upper_bound=10, fixed_pairs=None, 
               pre_processing=lambda x:x, post_processing=lambda y:y,
               model='meta-llama/Llama-2-7b-chat-hf', debug=False):
    int_as = []
    int_bs = []
    answers = []
    model_responses = []
    correct = []
    prompts = []
    iterations = range(n_sample) if fixed_pairs is None else fixed_pairs
    for i, v in enumerate(tqdm(iterations)):
        if fixed_pairs is None:
            int_a, int_b = get_addition_pairs(lower_bound=lower_bound, upper_bound=upper_bound, rng=rng)
        else:
            int_a, int_b = v
        fixed_prompt = f'{int_a}+{int_b}'
        fixed_prompt = pre_processing(fixed_prompt)
        prefix, suffix = added_prompt
        prompt = prefix + fixed_prompt + suffix
        model_response = call_together_api(prompt, prompt_configs, post_processing, model=model, debug=debug)
        answer = int_a + int_b
        int_as.append(int_a)
        int_bs.append(int_b)
        prompts.append(prompt)
        answers.append(answer)
        model_responses.append(model_response)
        correct.append((answer == model_response))
        sleep(1) # pause to not trigger DDoS defense
    df = pd.DataFrame({'int_a': int_as, 'int_b': int_bs, 'prompt': prompts, 'answer': answers, 'response': model_responses, 'correct': correct})
    print(df)
    mae = mean_absolute_error(df['answer'], df['response'])
    acc = df.correct.sum()/len(df)
    prompt_length = len(prefix) + len(suffix)
    res = acc * 1/prompt_length * (1-mae/(5*10**6))
    return {'res': res, 'acc': acc, 'mae': mae, 'prompt_length': prompt_length}

# %%
model_names = [
    "meta-llama/Llama-2-7b-chat-hf",  #LLaMa-2-7B
    "meta-llama/Llama-2-13b-chat-hf", #LLaMa-2-13B
    "meta-llama/Llama-2-70b-hf" #LLaMa-2-70B
]

# %%


# %% [markdown]
# **Example: Zero-shot single-digit addition**

# %%
added_prompt = ('Question: What is ', '?\nAnswer: ') # Question: What is a+b?\nAnswer:
prompt_config = {'max_tokens': 2,
                'temperature': 0.7,
                'top_k': 50,
                'top_p': 0.6,
                'repetition_penalty': 1,
                'stop': []}

# input_string: 'a+b'
def your_pre_processing(input_string):
    return input_string

# output_string: 
# depending on your prompt, it might look like 'output: number'
def your_post_processing(output_string):
    # using regular expression to find the first consecutive digits in the returned string
    only_digits = re.sub(r"\D", "", output_string)
    try:
        res = int(only_digits)
    except:
        res = 0
    return res

model = '"meta-llama/Llama-2-7b-chat-hf'
print(model)
seed = 0
rng = np.random.default_rng(seed)
res = test_range(added_prompt=added_prompt, prompt_configs=prompt_config, rng=rng, n_sample=10, lower_bound=1, upper_bound=10, fixed_pairs=None, pre_processing=your_pre_processing, post_processing=your_post_processing, model=model, debug=False)
print(res)

# %% [markdown]
# **Example: Zero-shot 7-digit addition**

# %%
sleep(1) # wait a little bit to prevent api call error
prompt_config['max_tokens'] = 8
rng = np.random.default_rng(seed)
res = test_range(added_prompt=added_prompt, prompt_configs=prompt_config, rng=rng, n_sample=10, lower_bound=1000000, upper_bound=9999999, fixed_pairs=None, pre_processing=your_pre_processing, post_processing=your_post_processing, model=model, debug=False)
print(res)

# %% [markdown]
# -----------

# %% [markdown]
# **Q1a.** In your opinion, what are some factors that cause language model performance to deteriorate from 1 digit to 7 digits?

# %% [markdown]
# Answer: 

# %% [markdown]
# -----------

