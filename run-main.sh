python caafe/main.py $1 $2 $3 2>&1 | grep -v -e 'warnings.warn' -e 'X, y, categorical_indicator, attribute_names = dataset.get_data(' -e 'FutureWarning' -e 'UserWarning'