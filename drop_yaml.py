import os
import json

# 定义要处理的目录列表
directories = [
    "/shared/ssd/models/fl-MED-SYN0-CLEVELAND/",
    "/shared/ssd/models/fl-MED-SYN0-HUNGARIAN/",
    "/shared/ssd/models/fl-MED-SYN0-SWITZERLAND/",
    "/shared/ssd/models/fl-MED-SYN0-VA/",
    "/shared/ssd/models/fl-MED-SYN1-CLEVELAND-train/",
    "/shared/ssd/models/fl-MED-SYN1-HUNGARIAN/",
    "/shared/ssd/models/fl-MED-SYN1-SWITZERLAND/",
    "/shared/ssd/models/fl-MED-SYN1-VA/"
]

# 要删除的字段
fields_to_remove = ["lora_bias", "exclude_modules", "eva_config"]

# 遍历每个目录
for directory in directories:
    config_path = os.path.join(directory, "adapter_config.json")
    
    # 检查文件是否存在
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = json.load(file)
        
        # 删除指定字段
        for field in fields_to_remove:
            if field in config_data:
                del config_data[field]
        
        # 将更新后的配置写回文件
        with open(config_path, 'w', encoding='utf-8') as file:
            json.dump(config_data, file, ensure_ascii=False, indent=4)
        
        print(f"Updated {config_path}")
    else:
        print(f"File not found: {config_path}")