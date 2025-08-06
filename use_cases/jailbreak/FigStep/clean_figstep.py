import os
import pandas as pd

# ———————— 用户修改区 ————————
INPUT_CSV   = "REBench/use_cases/jailbreak/FigStep/final_version_fitstep.csv"           # 原始 CSV 路径
OUTPUT_CSV  = "REBench/use_cases/jailbreak/FigStep/REBench_FigStep.csv"  # 处理后保存路径
COLUMN_NAME = "image_dir"          # 要裁剪的列名
# ——————————————————————————————

def main():
    # 1. 读取
    df = pd.read_csv(INPUT_CSV)

    # 2. 检查列是否存在
    if COLUMN_NAME not in df.columns:
        raise KeyError(f"列 {COLUMN_NAME!r} 不存在于 {INPUT_CSV} 中，可用列：{list(df.columns)}")

    # 3. 裁剪路径，只保留文件名
    df[COLUMN_NAME] = df[COLUMN_NAME].astype(str).apply(os.path.basename)

    # 4. 写入新文件
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 处理完毕，已将裁剪结果保存到：{OUTPUT_CSV}")

if __name__ == "__main__":
    main()