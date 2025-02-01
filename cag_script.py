# Import required libraries
import os
import platform
from time import time
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from sentence_transformers import SentenceTransformer
from helpers import get_env, generate_graphs


# Set the default device to "mps" for Mac M series chips
if platform.system() == "Darwin":
    torch.set_default_device("mps")
torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])
os.makedirs("data_cache", exist_ok=True)


# Define the CAGModule class
class CAGModule:
    def __init__(self, model_name: str, hf_token: str):
        self.model_name = model_name
        self.hf_token = hf_token
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token
        )
        print(f"Model: {model_name} loaded successfully.")

    def preprocess_knowledge(self, prompt: str) -> DynamicCache:
        embed_device = self.model.model.embed_tokens.weight.device
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(embed_device)
        past_key_values = DynamicCache()
        with torch.no_grad():
            outputs = self.model(
                input_ids = input_ids,
                past_key_values = past_key_values,
                use_cache = True,
                output_attentions = False,
                output_hidden_states = False
            )
        return outputs.past_key_values

    def write_kv_cache(self, kv: DynamicCache, path: str) -> None:
        torch.save(kv, path)

    def clean_up(self, kv: DynamicCache, origin_len: int) -> None:
        for i in range(len(kv.key_cache)):
            kv.key_cache[i] = kv.key_cache[i][:, :, :origin_len, :]
            kv.value_cache[i] = kv.value_cache[i][:, :, :origin_len, :]

    def prepare_kvcache(self, documents: str|list, kvcache_path: str, answer_instruction: str = None):
        if answer_instruction is None:
            answer_instruction = "Answer the question in a concise and precise way."

        if isinstance(documents, list):
            documents = '\n\n\n\n\n'.join(documents)
        elif isinstance(documents, str):
            pass
        else:
            raise ValueError("The `documents` parameter must be either a string or a list of strings.")

        knowledges = f"""
        <|start_of_role|>system<|end_of_role|>
        You are an assistant for giving precise answers based on given context.<|end_of_text|>
        <|start_of_role|>user<|end_of_role|>
        Context information is below.
        ------------------------------------------------
        {documents}
        ------------------------------------------------
        {answer_instruction}
        Question:
        """

        t1 = time()
        kv = self.preprocess_knowledge(knowledges)
        self.write_kv_cache(kv, kvcache_path)
        t2 = time()
        return kv, t2 - t1

    def generate(self, input_ids: torch.Tensor, past_key_values, max_new_tokens: int = 300):
        embed_device = self.model.model.embed_tokens.weight.device

        origin_ids = input_ids
        input_ids = input_ids.to(embed_device)

        output_ids = input_ids.clone()
        next_token = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                next_token_logits = outputs.logits[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1).to(embed_device)
                past_key_values = outputs.past_key_values

                output_ids = torch.cat([output_ids, next_token], dim=1)

                if next_token.item() == self.model.config.eos_token_id:
                    break
        return output_ids[:, origin_ids.shape[-1]:]

    def run_qna(self, question, knowledge_cache):
        prompt = f"""
            {question}<|end_of_text|>
            <|start_of_role|>assistant<|end_of_role|>
        """

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        output = self.generate(input_ids, knowledge_cache)
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

# Process multiple models on a given question-answer dataset
def run():
    HF_TOKEN = get_env()["HF_TOKEN"]
    datapath = "./datasets/rag_sample_qas_from_kis.csv"

    def get_kis_dataset(filepath):
        df = pd.read_csv(filepath)
        dataset = zip(df['sample_question'], df['sample_ground_truth'])
        text_list = df["ki_text"].to_list()
        return text_list, list(dataset)

    text_list, dataset = get_kis_dataset(datapath)

    model_names = [
        "ibm-granite/granite-3.0-2b-instruct",
        "ibm-granite/granite-3.1-2b-instruct",
        # "ibm-granite/granite-3.0-8b-instruct",
        # "ibm-granite/granite-3.1-8b-instruct"
    ]

    qa_details_per_model = {}
    model_summary_stats = {}

    bert_model = SentenceTransformer('all-MiniLM-L6-v2')

    for model_name in model_names:
        print(f"Processing model: {model_name}")
        model_id = model_name.replace("/", "_").replace(" ", "_")

        qa_details_per_model[model_id] = []
        model_summary_stats[model_id] = {}

        qna_module = CAGModule(model_name, HF_TOKEN)

        kv_cache_path = f"./data_cache/{model_id}_cache_knowledges.pt"
        knowledge_cache, prepare_time = qna_module.prepare_kvcache(text_list, kvcache_path=kv_cache_path)
        kv_len = knowledge_cache.key_cache[0].shape[-2]
        print("Length of the Key-Value (KV) Cache: ", kv_len)
        print(f"KV-Cache prepared in {prepare_time} seconds")

        total_similarity = 0
        total_inference_time = 0
        num_samples = len(dataset)

        for question, ground_truth in tqdm(dataset):
            torch.cuda.empty_cache()
            qna_module.clean_up(knowledge_cache, kv_len)

            generate_t1 = time()
            response = qna_module.run_qna(question=question, knowledge_cache=knowledge_cache)
            generate_t2 = time()

            response_time = generate_t2 - generate_t1
            total_inference_time += response_time

            ground_truth_emb = bert_model.encode(ground_truth, convert_to_tensor=True).cpu().numpy()
            response_emb = bert_model.encode(response, convert_to_tensor=True).cpu().numpy()
            similarity = cosine_similarity([ground_truth_emb], [response_emb])[0][0]
            total_similarity += similarity

            qa_details_per_model[model_id].append({
                "question": question,
                "ground_truth": ground_truth,
                "generated_text": response,
                "response_time": response_time,
                "similarity": similarity
            })

        avg_similarity = total_similarity / num_samples
        avg_inference_time = total_inference_time / num_samples

        model_summary_stats[model_id] = {
            "avg_similarity": avg_similarity,
            "avg_inference_time": avg_inference_time,
            "kv_len": kv_len,
            "prepare_time": prepare_time
        }

        del knowledge_cache
        del qna_module
        torch.cuda.empty_cache()

    return qa_details_per_model, model_summary_stats


# Run the script
if __name__ == "__main__":
    qa_details_per_model, model_summary_stats = run()
    generate_graphs(qa_details_per_model, model_summary_stats)