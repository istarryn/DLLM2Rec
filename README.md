# DLLM2Rec
### Introduction:
* This is the implementation of our paper: "Distillation Matters: Empowering Sequential Recommenders to Match the Performance of Large Language Models" (RecSys 2024).
* Our code is based on BIGRec (https://github.com/SAI990323/Grounding4Rec) and DROS (https://github.com/YangZhengyi98/DROS).
### Preparation:
Data processing and LLM-based Recommender in our work referred to BIGRec.
#### Data processing：
Our data processing approach is the same as BIGRec:
* Amazon_reviews: https://github.com/SAI990323/BIGRec/blob/main/data/game/process.ipynb
* MovieLens: https://github.com/SAI990323/BIGRec/blob/main/data/movie/process.ipynb
#### LLM-based Recommender：
To effectively utilize teacher's knowledge, please prepare following files from a LLM-based Recommender and save them at "./tocf/{dataset}":
* "all_embeddings.pt" (num_item X llm_emb_size) : the semantic embedding for all items encoded by LLM. (Please refer to "predict_embeddings" in https://github.com/SAI990323/BIGRec/blob/main/data/movie/evaluate.py)
* "myrank_train.txt"  (num_training_data X top_n) : the top-n ranking list for each training data from LLM-based Recommenders. (Please refer to "rank" in https://github.com/SAI990323/BIGRec/blob/main/data/movie/evaluate.py)
* "confidence_train.txt"  (num_training_data X top_n) : the top-n ranking confidence for each training data from LLM-based Recommenders. (Please refer to "dist" in https://github.com/SAI990323/BIGRec/blob/main/data/movie/evaluate.py)
### Run：
```
python main.py --data game --model_name SASRec --alpha 0.5 --ed_weight 0.3 --lam 0.7
```

