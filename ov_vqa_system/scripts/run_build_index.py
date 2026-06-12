import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_dataset


def build_formal_knowledge_base(max_articles=100000):
    print(f" 维基百科知识库构建 (目标规模: {max_articles} 篇文章)...")

    # 1. 从 Hugging Face 流式加载官方维基百科数据集 (20220301.en 版本)
    # 使用 streaming=True 可以防止内存爆炸，边下边处理
    print(" 正在连接 Hugging Face 获取 Wikipedia 语料...")
    dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

    # 2. 参数
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # 约等于 200 tokens
        chunk_overlap=200  # 约等于 50 tokens
    )

    passages = []
    print("✂️ 正在切分文章 (这可能需要一些时间)...")

    for i, article in enumerate(tqdm(dataset, total=max_articles, desc="解析文章")):
        if i >= max_articles:
            break
        # 过滤过短的文章
        if len(article['text']) < 500:
            continue

        chunks = text_splitter.split_text(article['text'])
        passages.extend(chunks)

    print(f" 语料切分完成，共获得 {len(passages)} 个 Passages。")

    # 3. 向量化编码
    print(" 加载 BGE-m3 模型并开始全量计算稠密向量...")
    embedder = SentenceTransformer('BAAI/bge-m3')

    # 为了防止显存溢出，设置 batch_size 并显示进度
    embeddings = embedder.encode(passages, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    embedding_dim = embeddings.shape[1]

    # 4. 构建并保存 FAISS 索引
    print("🗄️ 将向量灌入 FAISS 索引引擎...")
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    os.makedirs("./data/retrieval", exist_ok=True)
    faiss.write_index(index, "./data/retrieval/wikipedia_bge.index")

    # 保存文本映射
    print("正在保存文本映射字典...")
    with open("./data/retrieval/passages_mapping.json", "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False)

    print(" 正式版知识库构建大功告成！")


if __name__ == "__main__":
    # 正式实验时，可以把 100000 调大
    build_formal_knowledge_base(max_articles=100000)