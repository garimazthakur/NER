from transformers import TrainingArguments
from transformers.trainer_utils import EvaluationStrategy

TRN_PTH = "./resources/train.csv"
TST_PTH = "./resources/test.csv"
ANN_PTH = "./resources/annotation_data/"
DATASET_NAME = 'ANERCorp'
MODEL_NAME = 'aubmindlab/bert-base-arabertv02'
TASK_NAME = 'tokenclassification'
MODEL_PTH = './resources/best_model/'
RES_PTH = './resources/'


training_args = TrainingArguments("./resources/output")
training_args.evaluate_during_training = True
training_args.adam_epsilon = 1e-8
training_args.learning_rate = 5e-5
training_args.fp16 = True
training_args.per_device_train_batch_size = 8
training_args.per_device_eval_batch_size = 8
training_args.gradient_accumulation_steps = 2
training_args.num_train_epochs = 2
training_args.load_best_model_at_end = True
training_args.metric_for_best_model = True
training_args.metric_for_best_model = "eval_f1_score"
training_args.greater_is_better = True
training_args.resume_from_checkpoint = "./resources/output/"
# Warmup_ratio
warmup_ratio = 0.1

training_args.evaluation_strategy = EvaluationStrategy.EPOCH
training_args.logging_steps = 1
# training_args.save_strategy = "epoch"
training_args.save_steps = 3
training_args.seed = 42
training_args.disable_tqdm = False
training_args.lr_scheduler_type = 'cosine'
