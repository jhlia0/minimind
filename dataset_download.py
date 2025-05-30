from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="jhliao/minimind_dataset_zh_hant", repo_type="dataset", filename="pretrain_hq.jsonl", local_dir="./dataset")
hf_hub_download(repo_id="jhliao/minimind_dataset_zh_hant", repo_type="dataset", filename="sft_mini_512.jsonl", local_dir="./dataset")