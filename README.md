# DLLM2Rec

## Teacher model
We regard BIGRec(https://github.com/SAI990323/Grounding4Rec) as the LLM-based Recommender.
To effectively utilize teacher knowledge in following distillation, we save "item_embedding.pt", "ranking_list.txt" and "ranking_confidence.txt" for each dataset in the evaluation stage, where:
"item_embedding.pt" (num_item X 4096) : denotes the semantic embedding for all the items.
"ranking_list.txt"  (num_training_data X top_n) : denotes the top-n item ranking list for each training data.
"ranking_confidence.txt"  (num_training_data X top_n) : denotes the top-n item ranking confidence for each training data.

## Student model
Our code is based on DROS(https://github.com/YangZhengyi98/DROS).
### Requirements:
pytorch
numpy
pandas
nni==2.10.1
### Run models：
```
# w/ nni 
nnictl create --config config.yaml --port xxx
# w/o nni
python main.py --model_name SASRec --data game  
```

