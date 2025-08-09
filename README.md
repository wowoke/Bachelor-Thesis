Structure Tree

``` text
REBench
├── assembling.py
├── CITATION.cff
├── docs
│   └── source
│       ├── cli.rst
│       ├── conf.py
│       ├── index.rst
│       ├── installation.rst
│       ├── model.rst
│       └── usage.rst
├── LICENSE
├── MANIFEST.in
├── pyproject.toml
├── README.md
├── results # All results are here.
│   ├── baseline
│   ├── basemodel_results
│   ├── cognitive_overload_effect_to_cause
│   ├── cognitive_overload_veiled_attack
│   ├── FigStep
│   └── VRP-results
├── setup.cfg
├── src
│   ├── resp_gen_ai
│   │   ├── api.py
│   │   ├── cli.py
│   │   ├── constants.py
│   │   ├── datasets
│   │   ├── hydra_files
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── metrics
│   │   ├── models # Model settings are here.
│   │   ├── __pycache__
│   │   ├── py.typed
│   │   ├── tasks
│   │   └── version.py
│   └── resp_gen_ai.egg-info
├── structure.txt
├── tests
│   ├── __init__.py
│   └── test_version.py
├── tox.ini
└── use_cases
    ├── evalutor_llama_guard_3_8B.py # Evaluator llama guard entrance
    ├── evalutor_strong_reject_one_by_one.py # Evaluator strong reject entrance
    ├── jailbreak
    │   ├── baseline
    │   ├── cognitive_overload
    │   ├── FigStep
    │   ├── Multimodal_Pragmatic_Evaluation
    │   ├── run_models # Entrace of the repository
    │   └── VRP_query_specific
    └── load_inference
```
## 🛠 Environment Setup

Make sure you are using **Python 3.10.16**.

## 📦 Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/wowoke/REBench.git
cd REBench
pip install -e .
```

## ✅ Usage

After installation, you can start using the REBench toolkit.  
(You may add instructions here for running inference, benchmarking models, or specifying configuration.)

## 📂 Dataset Preparation

Before running the model, please make sure to download the dataset: **REBench_Dataset**.

In the Python script under `run_models` (e.g., `use_cases/jailbreak/run_models/jailbreak_vlm_R1_batch_vllm.py`),  
you need to set the following parameters before execution:

- The attack method you want to use (`jailbreak_method`)
- The path to the CSV data file (`csv_path`)
- The path to the image folder (`images_path`)

For example:

```python
csv_path = "REBench_Dataset/FigStep/REBench_FigStep.csv"
images_path = "REBench_Dataset/FigStep/FigStep_Images"
jailbreak_method = "FigStep"
```

Once set, run the script:

```bash
python use_cases/jailbreak/run_models/jailbreak_vlm_R1_batch_vllm.py
```

## 🧠 Running on Non-vLLM Models

For models that **do not support vLLM** (such as `Visual-RFT`, `Align-DSV`, `Open-R1-Multimodal`),  
you need to use the **`accelerate`** library, which should already be available in your environment.  
No extra installation is required.

Before launching, configure accelerate using:

```bash
accelerate config
```

This step will generate a `.yaml` configuration file under the default path.  
Based on your actual setup (e.g., number of GPUs), choose appropriate options.

Then launch the script using:

```bash
accelerate launch use_cases/jailbreak/run_models/jailbreak_pku_dsv_hf_batch.py
```

## 🚀 Talbe of Model Inference Method

| Model                              | Inference Method                                                                                       | Inference Mode |
|------------------------------------|--------------------------------------------------------------------------------------------------------|----------------|
| Align-DSV                          | accelerate launch use_cases/jailbreak/run_models/jailbreak_pku_dsv_hf_batch.py                         | accelerate     |
| QVQ-72B-Preview                    | python use_cases/jailbreak/run_models/jailbreak_qvq_72b_vllm_batch.py                                  | vLLM           |
| Skywork-R1V                        | python use_cases/jailbreak/run_models/jailbreak_skywork_R1V-vllm_batch.py                              | vLLM           |
| Visual-RFT                         | accelerate launch use_cases/jailbreak/run_models/jailbreak_visual_RFT_batch_hf.py                      | accelerate     |
| VLM-R1                             | accelerate launch use_cases/jailbreak/run_models/jailbreak_vlm_R1_batch_vllm.py                        | accelerate     |
| Open-R1-Multimodal                 | accelerate launch use_cases/jailbreak/run_models/jailbreak_open_r1_multimodal_hf_batch.py              | accelerate     |
| DeepSeek-R1-Distill-Llama-8B       | python use_cases/jailbreak/run_models/basemodels/deepseek-R1-Distill-Llama-8B.py                       | vLLM           |
| DeepSeek-R1-Distill-Qwen-32B       | python use_cases/jailbreak/run_models/basemodels/deepseek-R1-Distill-Qwen-32B.py                       | vLLM           |
| Qwen2-VL-7B-Instruct               | python use_cases/jailbreak/run_models/basemodels/qwen2-vl-7b-instruct-vllm.py                          | vLLM           |
| QVQ-72B-Preview                    | python use_cases/jailbreak/run_models/basemodels/qwen2-vl-72b-vllm.py                                  | vLLM           |
| Qwen2.5-VL-3B-Instruct             | python use_cases/jailbreak/run_models/basemodels/qwen2.5-vl-3B.py                                      | vLLM           |

## 🧪 How to Use the Judge Model?

To use only the **Judge Model**, simply specify the folder path where the model output results are stored in one of the following scripts:

- `REBench/use_cases/evalutor_llama_guard_3_8B.py`
- `REBench/use_cases/evalutor_strong_reject_one_by_one.py`

For example, in the script, modify the `folder_path` as follows:

```python
folder_path = ''

# redefine folder_path in the Python script
folder_path = 'models_results'
```

Once the path is correctly set, run one of the Python scripts below:

```bash
python use_cases/evalutor_llama_guard_3_8B.py
# or
python use_cases/evalutor_strong_reject_one_by_one.py
```

## 🔍 Pragmatic Evaluation Workflow

### 1. Clone the **Multimodal Pragmatic** repo and grab the image assets
```bash
git clone https://github.com/multimodalpragmatic/multimodalpragmatic.git
```
Save the **absolute path** to this `images` folder—you’ll paste it into the scripts below.

---

### 2. Build the zero-shot CSV
Edit  
`REBench/use_cases/jailbreak/Multimodal_Pragmatic_Evaluation/1_make_csv_file.py`  
and replace the placeholder on **line 27** with your `images` path.
Then run:
```bash
python REBench/use_cases/jailbreak/Multimodal_Pragmatic_Evaluation/1_make_csv_file.py
```
This creates `multimodalpragmatic_zero_shot.csv`.

---

### 3. Run your models
Open  
`use_cases/jailbreak/Multimodal_Pragmatic_Evaluation/2_run_code.py`  

* On **line 9**, point it to the CSV you just generated.  
* Select the model(s) you want to evaluate inside the script.

```bash
python REBench/use_cases/jailbreak/Multimodal_Pragmatic_Evaluation/2_run_code.py
```
The script outputs a **folder of JSON files**, one per model.

---

### 4. Check QWQ-32B alignment
Insert the path to that output folder in  
`use_cases/jailbreak/Multimodal_Pragmatic_Evaluation/3_evaluation_QWQ.py` (line 28) and run:

```bash
python use_cases/jailbreak/Multimodal_Pragmatic_Evaluation/3_evaluation_QWQ.py
```

---

### 5. Formalize the raw judgements
If the JSON schema still needs tweaking, edit  
`use_cases/jailbreak/Multimodal_Pragmatic_Evaluation/4_formalize_the_judgement.py` (line 5) with the file to be reformatted, then run:

```bash
python use_cases/jailbreak/Multimodal_Pragmatic_Evaluation/4_formalize_the_judgement.py
```

---

### 6. Compute the evaluation matrix
Add the **formalized JSON** path to  
`use_cases/jailbreak/Multimodal_Pragmatic_Evaluation/5_evaluation_matrix.py`, then execute:

```bash
python use_cases/jailbreak/Multimodal_Pragmatic_Evaluation/5_evaluation_matrix.py
```

The final script outputs a JSON file containing the **quantitative evaluation metrics** for each model.
