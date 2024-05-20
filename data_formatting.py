import json

# 等跑完了改成good
log_path = "/home/jiahe/ML/Self_instruct_CAAFE/caafe/log/good.jsonl"
# 等跑完了改成目标目录
output_path = "/home/jiahe/ML/Self_instruct_CAAFE/caafe/log/better.jsonl"

output_data=[]
with open(log_path,"r") as f:
    lines = f.readlines()
for line in lines:
    original_data = json.loads(line)
    formatted_data = {
        "messages": original_data["messages"]
    }
    formatted_data["messages"].append({
        "role": "assistant",
        "content": original_data["response"]
    })
    if float(original_data["improvement_roc"]) > 0 and float(original_data["improvement_acc"]) > 0 : 
        output_data.append(formatted_data)
with open(output_path,"w") as f:
    f.write(json.dumps(output_data,indent=2))