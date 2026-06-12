import os
import json
import string
from tqdm import tqdm
from datetime import datetime

from src.detector.yolo_wrapper import YoloWorldWrapper
from src.retrieval.faiss_retriever import FaissRetriever
from src.llm.ollama_wrapper import OllamaEngine
from src.pipelines.vqa_pipeline import OpenVocabVQAPipeline


def normalize_text(text):
    text = text.lower().strip()
    return text.translate(str.maketrans('', '', string.punctuation))


def run_aokvqa_evaluation():
    print("启动 A-OKVQA (多选题) 评测引擎...")

    # 初始化 Pipeline
    detector = YoloWorldWrapper(base_model_path='yolov8s-world.pt')
    retriever = FaissRetriever(
        index_path="./data/retrieval/wikipedia_bge.index",
        mapping_path="./data/retrieval/passages_mapping.json"
    )
    llm_engine = OllamaEngine(model_name="llama3.1")
    pipeline = OpenVocabVQAPipeline(detector, retriever, llm_engine, conf_threshold=0.5, top_k=5)

    # 加载 A-OKVQA 验证集
    dataset_path = "./data/raw/vqa/aokvqa_v1p0_val.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results, total_score, total_det, total_ev_len = [], 0.0, 0, 0

    for item in tqdm(dataset, desc="A-OKVQA 评测中"):
        img_name = item["image"]  # A-OKVQA 官方字段是 image (例如 "000000123456.jpg")
        img_path = os.path.join("../../data/common_images", img_name)

        question = item["question"]
        choices = item["choices"]
        correct_idx = item["correct_choice_idx"]
        correct_text = choices[correct_idx]

        # 给 LLM 的 Prompt 中附上选项，让它做选择题
        mc_question = f"{question} Choices: (A) {choices[0]} (B) {choices[1]} (C) {choices[2]} (D) {choices[3]}."

        ans, det_count, ev_len = pipeline.run(img_path, mc_question)

        # 严格匹配得分
        score = 1.0 if normalize_text(correct_text) in normalize_text(ans) else 0.0

        total_score += score
        total_det += det_count
        total_ev_len += ev_len

        results.append({
            "question_id": item["question_id"],
            "prediction": ans,
            "ground_truth": correct_text,
            "score": score
        })

    num = len(dataset)
    report = {
        "Dataset": "A-OKVQA Validation",
        "Accuracy": (total_score / num) * 100 if num > 0 else 0,
        "Avg_Retained_Detections": total_det / num if num > 0 else 0,
        "Avg_Evidence_Length": total_ev_len / num if num > 0 else 0
    }

    os.makedirs("../outputs", exist_ok=True)
    with open(f"./outputs/aokvqa_report_{datetime.now().strftime('%m%d_%H%M')}.json", "w") as f:
        json.dump({"metrics": report, "details": results}, f, indent=2)

    print(f"\n A-OKVQA 评测完成，准确率: {report['Accuracy']:.2f}%")


if __name__ == "__main__":
    run_aokvqa_evaluation()