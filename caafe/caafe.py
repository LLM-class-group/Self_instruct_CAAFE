import copy
import numpy as np

from client import chat_complete, get_time, print_important, get_time_in_float
from sklearn.model_selection import RepeatedKFold
from caafe_evaluate import (
    evaluate_dataset,
)
from run_llm_code import run_llm_code


def get_prompt(
    df, ds, iterative=1, data_description_unparsed=None, samples=None, **kwargs
):
    how_many = (
        "up to 10 useful columns. Generate as many features as useful for downstream classifier, but as few as necessary to reach good performance."
        if iterative == 1
        else "exactly one useful column"
    )
    return f"""
The dataframe `df` is loaded and in memory. Columns are also named attributes.
Description of the dataset in `df` (column dtypes might be inaccurate):
"{data_description_unparsed}"

Columns in `df` (true feature dtypes listed here, categoricals encoded as int):
{samples}
    
This code, written by an expert data scientist, adds new columns to the dataset to improve downstream classification (e.g., XGBoost) predicting "{ds[4][-1]}".
These columns incorporate real-world knowledge through feature combinations, transformations, and aggregations. Ensure all used columns exist. 
Consider datatypes and meanings of classes. The code also performs feature selection by dropping redundant columns to avoid overfitting. The classifier will be trained and evaluated on accuracy. Added columns can be reused, dropped columns cannot.

Code formatting for each added column:
```python
# (Feature name and description)
# Usefulness: (Description why this adds useful real world knowledge to classify \"{ds[4][-1]}\" according to dataset description and attributes.)
# Input samples: (Three samples of the columns used in the following code, e.g. '{df.columns[0]}': {list(df.iloc[:3, 0].values)}, '{df.columns[1]}': {list(df.iloc[:3, 1].values)}, ...)
(Some pandas code using {df.columns[0]}', '{df.columns[1]}', ... to add a new column for each row in df)
```end

Code formatting for dropping columns:
```python
# Explanation why the column XX is dropped
df.drop(columns=['XX'], inplace=True)
```end

Each codeblock generates {how_many} and can drop unused columns (Feature selection).
Each codeblock ends with ```end and starts with "```python"
Codeblock:
"""


# Each codeblock either generates {how_many} or drops bad columns (Feature selection).


def build_prompt_from_df(ds, df, iterative=1):
    data_description_unparsed = ds[-1]
    feature_importance = {}  # xgb_eval(_obj)

    samples = ""
    df_ = df.head(1)
    for i in list(df_):
        # show the list of values
        nan_freq = "%s" % float("%.2g" % (df[i].isna().mean() * 100))
        s = df_[i].tolist()
        if str(df[i].dtype) == "float64":
            s = [round(sample, 2) for sample in s]
        samples += (
            f"{df_[i].name} ({df[i].dtype}):Sample {s}\n"
        )

    kwargs = {
        "data_description_unparsed": data_description_unparsed,
        "samples": samples,
        "feature_importance": {
            k: "%s" % float("%.2g" % feature_importance[k]) for k in feature_importance
        },
    }

    prompt = get_prompt(
        df,
        ds,
        data_description_unparsed=data_description_unparsed,
        iterative=iterative,
        samples=samples,
    )

    return prompt


def generate_features(
    ds,
    df,
    model="gpt-3.5-turbo",
    just_print_prompt=False,
    iterative=1,
    metric_used=None,
    iterative_method="logistic",
    display_method="markdown",
    n_splits=10,
    n_repeats=2,
    rewrite_prompt = False
):
    def format_for_display(code):
        code = code.replace("```python", "").replace(
            "```", "").replace("<end>", "")
        return code

    if display_method == "markdown":
        from IPython.display import display, Markdown

        def display_method(x): return display(Markdown(x))
    else:

        display_method = print

    assert (
        iterative == 1 or metric_used is not None
    ), "metric_used must be set if iterative"
    
    if rewrite_prompt :
        data_description = ds[-1]
        simlify_messages = [
            {
            "role": "system",
            "content": "You are a helpful assistant . You will simplify the data description,only preserve those which is critical for feature engineering. ",
            },
            {
            "role": "user",
            "content": "You are a helpful assistant . You will simplify the data description,only preserve those which is critical for feature engineering. You should only response the simplified description .Your answer should begin with \"```begin\" .Dataset description:\n"+data_description,
            },
        ]

        _, simplified_description = chat_complete(
            messages=simlify_messages,
            stop=["```end"],
            temperature=0.5,
            max_tokens=500,
        )
        simplified_description = simplified_description.replace("```begin", "").replace(
            "```", "").replace("<end>", "")
        ds[-1] = simplified_description
        
    prompt = build_prompt_from_df(ds, df, iterative=iterative)

    if just_print_prompt:
        code, prompt = None, prompt
        return code, prompt, None

    def generate_code(messages):
        if model == "skip":
            return ""

        # completion = openai.ChatCompletion.create(
        #     model=model,
        #     messages=messages,
        #     stop=["```end"],
        #     temperature=0.5,
        #     max_tokens=500,
        # )
        # code = completion["choices"][0]["message"]["content"]

        completion, code = chat_complete(
            messages=messages,
            stop=["```end"],
            temperature=0.5,
            max_tokens=500,
        )

        # code = completion.choices[0].message.content

        # --------------------------------------------------------

        code = code.replace("```python", "").replace(
            "```", "").replace("<end>", "")
        return code

    def execute_and_evaluate_code_block(full_code, code):
        old_accs, old_rocs, accs, rocs = [], [], [], []

        ss = RepeatedKFold(n_splits=n_splits,
                           n_repeats=n_repeats, random_state=0)
        for (train_idx, valid_idx) in ss.split(df):
            df_train, df_valid = df.iloc[train_idx], df.iloc[valid_idx]

            # Remove target column from df_train
            target_train = df_train[ds[4][-1]]
            target_valid = df_valid[ds[4][-1]]
            df_train = df_train.drop(columns=[ds[4][-1]])
            df_valid = df_valid.drop(columns=[ds[4][-1]])

            df_train_extended = copy.deepcopy(df_train)
            df_valid_extended = copy.deepcopy(df_valid)

            try:
                df_train = run_llm_code(
                    full_code,
                    df_train,
                    convert_categorical_to_integer=not ds[0].startswith(
                        "kaggle"),
                )
                df_valid = run_llm_code(
                    full_code,
                    df_valid,
                    convert_categorical_to_integer=not ds[0].startswith(
                        "kaggle"),
                )
                df_train_extended = run_llm_code(
                    full_code + "\n" + code,
                    df_train_extended,
                    convert_categorical_to_integer=not ds[0].startswith(
                        "kaggle"),
                )
                df_valid_extended = run_llm_code(
                    full_code + "\n" + code,
                    df_valid_extended,
                    convert_categorical_to_integer=not ds[0].startswith(
                        "kaggle"),
                )

            except Exception as e:
                display_method(f"Error in code execution. {type(e)} {e}")
                display_method(f"```python\n{format_for_display(code)}\n```\n")
                return e, None, None, None, None, None

            # Add target column back to df_train
            df_train[ds[4][-1]] = target_train
            df_valid[ds[4][-1]] = target_valid
            df_train_extended[ds[4][-1]] = target_train
            df_valid_extended[ds[4][-1]] = target_valid

            from contextlib import contextmanager
            import sys
            import os

            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    result_old = evaluate_dataset(
                        df_train=df_train,
                        df_test=df_valid,
                        name=ds[0],
                        method=iterative_method,
                        metric_used=metric_used,
                        seed=0,
                        target_name=ds[4][-1],
                    )

                    result_extended = evaluate_dataset(
                        df_train=df_train_extended,
                        df_test=df_valid_extended,
                        name=ds[0],
                        method=iterative_method,
                        metric_used=metric_used,
                        seed=0,
                        target_name=ds[4][-1],
                    )
                finally:
                    sys.stdout = old_stdout

            old_accs += [result_old["acc"]]
            old_rocs += [result_old["roc"]]
            accs += [result_extended["acc"]]
            rocs += [result_extended["roc"]]
        # also return new df
        return None, rocs, accs, old_rocs, old_accs, copy.deepcopy(df_train)

    messages = [
        {
            "role": "system",
            "content": "You are an expert datascientist assistant solving Kaggle problems. You answer only by generating code. Answer as concisely as possible.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    display_method(f"*Dataset description:*\n {ds[-1]}")

    n_iter = iterative
    full_code = ""

    i = 0
    success = 0
    query = 0
    # pure_messages: exclude error messages
    pure_messages = messages
    while i < n_iter:
        time_0 = get_time_in_float()
        print_important(
            f"----- Start feature iteration {i+1} at {get_time()} -----")

        try:
            query = query + 1
            code = generate_code(messages)
        except Exception as e:
            display_method("Error in LLM API." + str(e))
            continue
        i = i + 1

        time_1 = get_time_in_float()
        print_important(
            f"feature generation spend time: {float(time_1) - float(time_0)}")

        e, rocs, accs, old_rocs, old_accs, df_extended = execute_and_evaluate_code_block(
            full_code, code
        )
        if e is not None:
            messages = pure_messages + [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""Code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```\n Generate another feature.Do not output any other words.
Next codeblock:
""",
                },
            ]
            continue
        # importances = get_leave_one_out_importance(
        #    df_train_extended,
        #    df_valid_extended,
        #    ds,
        #    iterative_method,
        #    metric_used,
        # )
        # """ROC Improvement by using each feature: {importances}"""

        improvement_roc = np.nanmean(rocs) - np.nanmean(old_rocs)
        improvement_acc = np.nanmean(accs) - np.nanmean(old_accs)

        time_2 = get_time_in_float()
        print_important(
            f"execute code and evaluate spend time: {float(time_2) - float(time_1)}")

        #
        # ------------------------------
        # log good results
        # ------------------------------
        #
        log_path = "/home/jiahe/ML/Self_instruct_CAAFE/caafe/log/good2.jsonl"

        if improvement_roc > 0.001 and improvement_acc > 0.001:
            print_important(f"!! Log one good response from LLM !!")
            success = success + 1
            log_messages = [
                {
                    "role": "system",
                    "content": "You are an expert datascientist assistant solving Kaggle problems. You answer only by generating code. Answer as concisely as possible.",
                },
                {
                    "role": "user",
                    "content": build_prompt_from_df(ds, df_extended, iterative=iterative),
                },
            ]

            log_entry = {
                "messages": log_messages,
                "response": code,
                "improvement_roc": improvement_roc,
                "improvement_acc": improvement_acc,
            }

            try:
                with open(log_path, "a") as log_file:
                    import json
                    json.dump(log_entry, log_file)
                    log_file.write("\n")  # Add newline to separate entries
            except Exception as e:
                print(f"Warning: Could not write to log file: {e}")
        #
        # ------------------------------
        # ------------------------------
        #

        add_feature = True
        add_feature_sentence = "The code was executed and changes to ´df´ were kept."
        if improvement_roc + improvement_acc <= 0:
            add_feature = False
            add_feature_sentence = f"The last code changes to ´df´ were discarded. (Improvement: {improvement_roc + improvement_acc})"

        display_method(
            "\n"
            + f"```python\n{format_for_display(code)}\n```\n"
            + f"Performance before adding features ROC {np.nanmean(old_rocs):.3f}, ACC {np.nanmean(old_accs):.3f}.\n"
            + f"Performance after adding features ROC {np.nanmean(rocs):.3f}, ACC {np.nanmean(accs):.3f}.\n"
            + f"Improvement ROC {improvement_roc:.3f}, ACC {improvement_acc:.3f}.\n"
            + f"{add_feature_sentence}\n"
            + f"\n"
        )
        if len(code) > 10:
            messages = pure_messages + [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""Performance after adding feature ROC {np.nanmean(rocs):.3f}, ACC {np.nanmean(accs):.3f}. {add_feature_sentence}
Next codeblock:
""",
                },
            ]
        if add_feature:
            full_code += code
        pure_messages = messages

    # finished generation_features
    print_important(
        f"#### cycle finished with {success} good codes out of {query} queries at {get_time()}. Success rate: {success/query} #####")

    return full_code, prompt, messages
