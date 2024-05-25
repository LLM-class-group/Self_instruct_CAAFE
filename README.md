# Self-Instruct CAAFE

Our project is a combination of CAAFE’s basic method and the innovative Self-Instruct method.

### 项目结构

- `caafe/`: 代码目录
- `data/`: 数据目录，其中`eval_*.txt`记录了详细评估结果，`finetuning_data.json`是我们最终微调使用的数据
- `evaluation.sh`: 评估脚本
- `finetune.sh`：微调脚本
- `run-main.sh`：生成数据脚本
- `server.sh` & `server-sft.sh` : vllm脚本

### 评估结果 

seed = 42 & 24, metric = AUC

| Method        | No Feat. Eng. | Llama3-8B (Iter=3) | Llama3-8B (Iter=10) | Llama3-8B-SFT (Iter=3) | Llama3-8B-SFT (Iter=10) |
| ------------- | ------------- | ------------------ | ------------------- | ---------------------- | ----------------------- |
| TabPFN        | 0.8117        | 0.8189 ± 0.0059    | 0.8232 ± 0.0045     | 0.8447 ± 0.0055        | 0.8595 ± 0.0093         |
| Random Forest | 0.7836        | 0.8219 ± 0.0068    | 0.8167 ± 0.0115     | 0.8403 ± 0.0019        | 0.8561 ± 0.0038         |
| XGBoost       | 0.7790        | 0.7862 ± 0.0052    | 0.8136 ± 0.0109     | 0.8039 ± 0.0054        | 0.8221 ± 0.0093         |

### 流程图
![Process Diagram](process.png)

