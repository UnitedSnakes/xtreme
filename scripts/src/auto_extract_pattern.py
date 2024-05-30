import pandas as pd

from parameters import *
from utils import *

def extract_patterns(language):
    df_raw = pd.read_csv(Config.results_file.replace(".csv", f"_{language}.csv"))
    # print(df_raw.dtypes)
    # input()

    df_warnings = check_prediction_format(df_raw, extract=True, verbose=False)
    # print(df_raw.dtypes)
    # print(df_warnings.dtypes)
    # input()
    
    # for i in df_warnings["prediction"]:
    #     print(type(i))
    # input()
    
    # print(df_raw.shape[0])

    # Merge the two DataFrames on 'id'
    df_merged = df_raw.merge(df_warnings[['id', 'prediction']], on='id', how='left', suffixes=('', '_warning'))
    # print(df_merged.shape[0])
    # input()
    # Update the prediction column in df_raw with values from df_warnings
    df_merged['prediction'] = df_merged['prediction_warning'].combine_first(df_merged['prediction'])
    # print(df_merged.shape[0])
    # input()
    # for i in df_merged["prediction"]:
    #     print(type(i))
    # print(language)
    # input()
    # print(df_merged.dtypes)
    # input()

    # Drop the extra column
    df_merged.drop(columns=['prediction_warning'], inplace=True)
    # print(df_merged.shape[0])
    # input()
    print(check_prediction_format(df_merged, verbose=False))
    
    # for i in df_raw["prediction"]:
    #     if type(i) != str:
    #         print(i)
    #         print(type(i))
    # print(language)
    # input()
    # print(df_merged.dtypes)
    # input()
    # print(df_merged.shape[0])
    # input()
    df_merged.to_csv(Config.annotated_results_file.replace(".csv", f"_{language}.csv"))


def main():
    for language in Config.evaluate_languages:
        extract_patterns(language)
    
if __name__ == '__main__':
    main()
