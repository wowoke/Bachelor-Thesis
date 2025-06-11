Structure Tree

``` text
.
├── CITATION.cff
├── LICENSE
├── MANIFEST.in
├── README.md
├── assembling.py
├── docs
│   └── source
├── outputs
│   └── 2025-03-16
├── pyproject.toml
├── setup.cfg
├── src
│   ├── resp_gen_ai
│   └── resp_gen_ai.egg-info
├── tests
│   ├── __init__.py
│   └── test_version.py
├── tox.ini
└── use_cases
    ├── calculate_asr_lm.py
    ├── calculate_asr_lm_for_single.py
    ├── calculate_only_asr_from_json.py
    ├── evaluation_results
    ├── evalutor_llama_guard_3_8B.py
    ├── evalutor_strong_reject.py
    ├── evalutor_strong_reject_one_by_one.py
    ├── evalutor_strong_reject_one_by_one_copy.py
    ├── hallucination
    ├── jailbreak
    ├── load_inference
    ├── memorization
    └── reasoning 
```
These files specify the parameters for loading the model and the parameters used during generation. However, these parameters are not applicable to vLLM.

``` bash
RespGenAI-zhongyi/src/resp_gen_ai/models
```
This is where the models above are called. In addition, the code in these places loads databases and slices them into batch and passes them into the model, then saves the results as a json file.
``` bash
RespGenAI-zhongyi/use_cases/jailbreak/truly_usefull_codes
```
The place to store the results is here. Please ignore files outside of folders, the official results are in folders. The folders are named after the jailbreak attack method. The new_dsv is an exception, it's another version of Align-ds-v that I ran.
``` bash
RespGenAI-zhongyi/use_cases/evaluation_results
```
The code that loads the judge model is the following two scripts, which read the output of the model and load it into the judge model one by one, and then save the evaluaiton to the result in the folder above.
``` bash
RespGenAI-zhongyi/use_cases/evalutor_llama_guard_3_8B.py
RespGenAI-zhongyi/use_cases/evalutor_strong_reject_one_by_one.py
```
As for the datasets. They are all in here, please ignore the codes outside the folders, the formal dataset used in this work are all in the folder named by the jailbreak methods, and they are all csv files.
``` bash
RespGenAI-zhongyi/use_cases/jailbreak
```
With the large number of images used in VRP and FigStep, I didn't find an elegant way to upload them because they were too big, 21 GB， so I decided to give up on uploading them.
Environment configuration information can be found in pyproject.toml.
