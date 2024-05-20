import re

old_long = """This code was written by an expert datascientist working to improve predictions. It is a snippet of code that adds new columns to the dataset.
Number of samples (rows) in training dataset: {int(len(df))}
    
This code generates additional columns that are useful for a downstream classification algorithm (such as XGBoost) predicting \"{ds[4][-1]}\".
Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns.
The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of columns closely and consider the datatypes and meanings of classes.
This code also drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier (Feature selection). Dropping columns may help as the chance of overfitting is lower, especially if the dataset is small.
The classifier will be trained on the dataset with the generated columns and evaluated on a holdout set. The evaluation metric is accuracy. The best performing code will be selected.
Added columns can be used in other codeblocks, dropped columns are not available anymore."""

new_long = """This code, written by an expert data scientist, adds new columns to the dataset to improve downstream classification (e.g., XGBoost) predicting "{ds[4][-1]}".
These columns incorporate real-world knowledge through feature combinations, transformations, and aggregations. Ensure all used columns exist. 
Consider datatypes and meanings of classes. The code also performs feature selection by dropping redundant columns to avoid overfitting. The classifier will be trained and evaluated on accuracy. Added columns can be reused, dropped columns cannot."""


def extract_parts(input_string):
    # 正则表达式来匹配 data_description_unparsed 的值
    begin_string = "Description of the dataset in `df` (column dtypes might be inaccurate):\n\""
    end_string = "\"\n\nColumns in `df`"
    #data_description_pattern = r'Description of the dataset in `df` \(column dtypes might be inaccurate\):\n"((?:.|\n)*?)"\n\nColumns in `df`'
    #data_description_match = re.search(data_description_pattern, input_string)
    #data_description = data_description_match.group(
    #    1) if data_description_match else None
    after_begin = input_string.split(begin_string)[1]
    data_description = after_begin.split(end_string)[0]
    # 正则表达式来匹配 samples 的值
    samples_pattern = r'Columns in `df` \(true feature dtypes listed here, categoricals encoded as int\):\n([^}]+)'
    samples_match = re.search(samples_pattern, input_string)
    samples = samples_match.group(1) if samples_match else None
    return data_description, samples


def update_long(old_prompt):
    # 替换一大坨内容
    new_prompt = re.sub(old_long, new_long, old_prompt)
    return new_prompt
