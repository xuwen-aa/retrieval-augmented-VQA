from src.detector.yolo_wrapper import YoloWorldWrapper
from src.pipelines.vqa_pipeline import OpenVocabVQAPipeline
from src.retrieval.faiss_retriever import FaissRetriever

# --- 模拟组件 (后续替换为真实 API) ---
class DummyRetriever:
    def search(self, query, top_k=3):
        return ["stop sign", "pedestrian", "bicycle"]


class DummyLLMEngine:
    def generate(self, prompt):
        return "Based on the detection of a stop sign and pedestrians, the vehicle should stop."


# ------------------------------------

def main():
    # 实例化 YOLO-World
    detector = YoloWorldWrapper(base_model_path='yolov8s-world.pt')

    # 实例化真实的 FAISS 检索器
    # 注意：运行前请确保你已经跑过 run_build_index.py 生成了这两个文件！
    retriever = FaissRetriever(
        index_path="./data/retrieval/wikipedia_bge.index",
        mapping_path="./data/retrieval/passages_mapping.json"
    )

    # 实例化 LLaMA-3 (Ollama)
    llm_engine = OllamaEngine(model_name="llama3.1")

    # 组装超级 Pipeline
    pipeline = OpenVocabVQAPipeline(detector, retriever, llm_engine)

    # 3. 运行测试
    # 找一张你电脑里随便的图片测试一下链路 (随便给个有效路径即可)
    test_image = "test.jpg"
    test_question = "What traffic regulations apply to this scene?"

    pipeline.run(test_image, test_question)


if __name__ == "__main__":
    main()