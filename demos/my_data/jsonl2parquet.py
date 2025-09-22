import pandas as pd
import json
import argparse

def jsonl_to_parquet(jsonl_file_path, parquet_file_path):
    """
    将JSONL文件转换为Parquet文件。

    Args:
        jsonl_file_path (str): 输入的JSONL文件路径。
        parquet_file_path (str): 输出的Parquet文件路径。
    """
    try:
        # 读取JSONL文件
        # 可以通过lines=True直接读取，或者逐行读取并解析
        # 如果JSONL文件中的每一行都是一个完整的JSON对象，可以直接使用pd.read_json
        df = pd.read_json(jsonl_file_path, lines=True)

        # 另一种更健壮的逐行读取方式，以防某些行格式不标准
        # data = []
        # with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         try:
        #             data.append(json.loads(line.strip()))
        #         except json.JSONDecodeError as e:
        #             print(f"Warning: Skipping malformed JSON line: {line.strip()} - {e}")
        # if not data:
        #     print("Error: No valid JSON data found in the file.")
        #     return
        # df = pd.DataFrame(data)


        # 将DataFrame保存为Parquet文件
        df.to_parquet(parquet_file_path, engine='pyarrow', index=False)
        print(f"Successfully converted '{jsonl_file_path}' to '{parquet_file_path}'")

    except FileNotFoundError:
        print(f"Error: JSONL file not found at '{jsonl_file_path}'")
    except pd.errors.EmptyDataError:
        print(f"Error: JSONL file '{jsonl_file_path}' is empty or contains no valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # 示例用法

#     # 1. 创建一个示例JSONL文件 (如果还没有的话)
#     sample_jsonl_content = """
# {"id": 1, "name": "Alice", "age": 30, "city": "New York"}
# {"id": 2, "name": "Bob", "age": 24, "city": "Los Angeles"}
# {"id": 3, "name": "Charlie", "age": 35, "city": "Chicago", "occupation": "Engineer"}
# {"id": 4, "name": "David", "age": 29, "city": "Houston"}
# """
#     with open("sample.jsonl", "w", encoding="utf-8") as f:
#         f.write(sample_jsonl_content.strip())
#     print("Created 'sample.jsonl'")

    # 2. 调用转换函数
    parser = argparse.ArgumentParser(description="Convert JSONL to Parquet")
    parser.add_argument("--jsonl", type=str, default="demo.jsonl", help="\
                        Path to the input JSONL file")
    parser.add_argument("--parquet", type=str, default="demo.parquet", help="\
                        Path to the output Parquet file")
    args = parser.parse_args()
    jsonl_file = args.jsonl
    parquet_file = args.parquet
    jsonl_to_parquet(jsonl_file, parquet_file)

    # 3. 验证 Parquet 文件 (可选)
    try:
        print("\nVerifying the created Parquet file:")
        df_parquet = pd.read_parquet(parquet_file)
        print(df_parquet.head())
        print(f"Shape of Parquet data: {df_parquet.shape}")
    except Exception as e:
        print(f"Error verifying Parquet file: {e}")
