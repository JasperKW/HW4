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

model = 'meta-llama/Llama-2-7b-chat-hf'
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

# %% [markdown]
# **Q1b**. Play around with the config parameters ('max_tokens','temperature','top_k','top_p','repetition_penalty') in together.ai's [web UI](https://api.together.xyz/playground/language/togethercomputer/llama-2-7b). 
# * What does each parameter represent?
# * How does increasing each parameter change the generation?

# %% [markdown]
# Answer: 

# %% [markdown]
# -----------

# %% [markdown]
# **Q1c**. Do 7-digit addition with 70B parameter llama model. 
# * How does the performance change?
# * What are some factors that cause this change?

# %% [markdown]
# Answer: 

# %%
sleep(1) # wait a little bit to prevent api call error
rng = np.random.default_rng(seed)
res = test_range(added_prompt=added_prompt, prompt_configs=prompt_config, rng=rng, n_sample=10, lower_bound=1000000, upper_bound=9999999, fixed_pairs=None, pre_processing=your_pre_processing, post_processing=your_post_processing, model='meta-llama/Llama-2-70b-hf', debug=False)
print(res)

# %% [markdown]
# -----------

# %% [markdown]
# **Q1d.** Here we're giving our language model the prior that the sum of two 7-digit numbers must have a maximum of 8 digits. (by setting max_token=8). What if we remove this prior by increasing the max_token to 20? 
# * Does the model still perform well?
# * What are some reasons why?

# %% [markdown]
# Answer: 

# %%
sleep(1) # wait a little bit to prevent api call error
added_prompt = ('Question: What is ', '?\nAnswer: ') # Question: What is a+b?\nAnswer:
prompt_config = {'max_tokens': 20,
                'temperature': 0.7,
                'top_k': 50,
                'top_p': 0.6,
                'repetition_penalty': 1,
                'stop': []}

# input_string: 'a+b'
def your_pre_processing(input_string):
    return input_string

def your_post_processing(output_string):
    first_line = output_string.splitlines()[0]
    only_digits = re.sub(r"\D", "", first_line)
    try:
        res = int(only_digits)
    except:
        res = 0
    return res


model = 'meta-llama/Llama-2-7b-chat-hf'
print(model)
seed = 0
rng = np.random.default_rng(seed)
res = test_range(added_prompt=added_prompt, prompt_configs=prompt_config, rng=rng, n_sample=10, lower_bound=1000000, upper_bound=9999999, fixed_pairs=None, pre_processing=your_pre_processing, post_processing=your_post_processing, model=model, debug=False)
print(res)

# %% [markdown]
# ### Part 2. In Context Learning
# 
# We will try to improve the performance of 7-digit addition via in-context learning.
# For cost-control purposes (you only have $25 free credits), we will use [llama-2-7b](https://api.together.xyz/playground/language/togethercomputer/llama-2-7b). Below is a simple example.

# %%
sleep(1) # wait a little bit to prevent api call error
added_prompt = ('Question: What is 3+7?\nAnswer: 10\n Question: What is ', '?\nAnswer: ') # Question: What is a+b?\nAnswer:
prompt_config = {'max_tokens': 8,
                'temperature': 0.7,
                'top_k': 50,
                'top_p': 0.6,
                'repetition_penalty': 1,
                'stop': []}
rng = np.random.default_rng(seed)
res = test_range(added_prompt=added_prompt, prompt_configs=prompt_config, rng=rng, n_sample=10, lower_bound=1000000, upper_bound=9999999, fixed_pairs=None, pre_processing=your_pre_processing, post_processing=your_post_processing, model='meta-llama/Llama-2-7b-chat-hf', debug=False)
print(res)

# %% [markdown]
# **Q2a**.
# * How does the performance change with the baseline in-context learning prompt? (compare with "Example: Zero-shot 7-digit addition" in Q1)
# * What are some factors that cause this change?

# %% [markdown]
# Answer: 

# %% [markdown]
# -----------

# %% [markdown]
# Now we will remove the prior on output length and re-evaluate the performance of our baseline one-shot learning prompt. We need to modify our post processing function to extract the answer from the output sequence. In this case, it is the number in the first line that starts with "Answer: ".

# %% [markdown]
# **Q2b**.
# * How does the performance change when we relax the output length constraint? (compare with Q2a)
# * What are some factors that cause this change?

# %% [markdown]
# Answer: 

# %%
sleep(1) # wait a little bit to prevent api call error

prompt_config['max_tokens'] = 50 # changed from 8, assuming we don't know the output length
                
rng = np.random.default_rng(seed)
res = test_range(added_prompt=added_prompt, prompt_configs=prompt_config, rng=rng, n_sample=10, lower_bound=1000000, upper_bound=9999999, fixed_pairs=None, pre_processing=your_pre_processing, post_processing=your_post_processing, model='meta-llama/Llama-2-7b-chat-hf', debug=False)
print(res)

# %% [markdown]
# -----------

# %% [markdown]
# **Q2c.** Let's change our one-shot learning example to something more "in-distribution". Previously we were using 1-digit addition as an example. Let's change it to 7-digit addition (1234567+1234567=2469134). 
# * Evaluate the performance with max_tokens = 8.
# * Evaluate the performance with max_tokens = 50.
# * How does the performance change from 1-digit example to 7-digit example?

# %% [markdown]
# Answer: 

# %%
sleep(1) # wait a little bit to prevent api call error
prompt_config['max_tokens'] = 8 
added_prompt = ('Question: What is 1234567+123457?\nAnswer: 2469134\nQuestion: What is ', '?\nAnswer: ') # Question: What is a+b?\nAnswer:
test_range(added_prompt=added_prompt, prompt_configs=prompt_config, rng=rng, n_sample=10, lower_bound=1000000, upper_bound=9999999, fixed_pairs=None, pre_processing=your_pre_processing, post_processing=your_post_processing, model='meta-llama/Llama-2-7b-chat-hf', debug=False)

# %%
sleep(1) # wait a little bit to prevent api call error
prompt_config['max_tokens'] = 50 
test_range(added_prompt=added_prompt, prompt_configs=prompt_config, rng=rng, n_sample=10, lower_bound=1000000, upper_bound=9999999, fixed_pairs=None, pre_processing=your_pre_processing, post_processing=your_post_processing, model='meta-llama/Llama-2-7b-chat-hf', debug=False)

# %% [markdown]
# -----------

# %% [markdown]
# **Q2d.** Let's look at a specific example with large absolute error. 
# * Run the cell at least 5 times. Does the error change each time? Why?
# * Can you think of a prompt to reduce the error?
# * Why do you think it would work?
# * Does it work in practice? Why or why not?

# %% [markdown]
# Answer:

# %%
test_range(added_prompt=added_prompt, prompt_configs=prompt_config, rng=rng, fixed_pairs=[(9090909,1010101)], pre_processing=your_pre_processing, post_processing=your_post_processing, model='meta-llama/Llama-2-7b-chat-hf', debug=True)

# %% [markdown]
# ### Part 3: Prompt-a-thon (autograder & leaderboard)
# 

# %% [markdown]
# Compete with your classmates to see who's best at teach llama to add 7-digit numbers reliably! Submit your ```submission.py``` to enter the leader board!
# 
# Note: while you can use prompt.txt for debugging and local testing, for the final autograder submission, please use a string (not a file), because autograder cannot find prompt.txt in the testing environment. Sorry about the inconvenience!
# 
# What you can change:
# * your_api_key
# * your_prompt
# * your_config
# * your_pre_processing
# * your_post_processing

# %% [markdown]
# 


