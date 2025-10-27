#!/usr/bin/env python3
import sys
import os
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading model...", file=sys.stderr)

local_model_path = os.path.expanduser("../models/paraphrase-multilingual-mpnet-base-v2-local")

try:
    if os.path.exists(local_model_path):
        print(f"Loading from local path: {local_model_path}", file=sys.stderr)
        model = SentenceTransformer(local_model_path)
    else:
        print("Local model not found, using model name...", file=sys.stderr)
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    sys.exit(1)

print("Model loaded successfully!", file=sys.stderr)

while True:
    try:
        line = sys.stdin.readline()
        if not line or line.strip() == "EXIT":  # 添加EXIT检查
            break
            
        texts = line.strip().split("|||")
        
        # 批量编码
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        # 输出每个归一化向量（一行一个）
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            if norm > 0:
                normalized_emb = emb / norm
            else:
                normalized_emb = emb
            
            # 输出为空格分隔的浮点数
            print(' '.join(map(str, normalized_emb)))
            sys.stdout.flush()
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        break  # 出错时也退出