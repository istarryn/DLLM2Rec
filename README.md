# DLLM2Rec
### Introduction:
This is the implementation of our paper: "Distillation Matters: Empowering Sequential Recommenders to Match the Performance of Large Language Models".
### Preparation:
Data processing and LLM-based Recommender in our work referred to https://github.com/SAI990323/Grounding4Rec.

To effectively utilize teacher's knowledge, please prepare following files from a LLM-based Recommender and save them at "./tocf/{dataset}":
* "all_embeddings.pt" (num_item X llm_emb_size) : the semantic embedding for all items encoded by LLM.
* "myrank_train.txt"  (num_training_data X top_n) : the top-n ranking list for each training data from LLM-based Recommenders.
* "confidence_train.txt"  (num_training_data X top_n) : the top-n ranking confidence for each training data from LLM-based Recommenders.
### Run：
```
python main.py --data game --model_name SASRec --alpha 0.5 --ed_weight 0.3 --lam 0.7
```

