# Text-to-SQL: QLoRA Fine-Tuning of Llama 3.2 3B
 
Fine-tuning of **Llama 3.2 3B Instruct** with **QLoRA** Unsloth so that, given a **table schema** and a **question in natural language**, the model writes the matching **SQL query** and nothing else.
 
The fine-tuned model is compared with the base model on 300 held-out examples. It produces a query that runs against the schema in **99.3%** of cases (base model: 91.7%) and finishes its generation about **6 times faster**.
 

## Results
 
Evaluation on 300 test examples whose schemas and questions never appear in training.
 
| Model | Valid queries | Generation time (300 queries) |
|---|---|---|
| Llama 3.2 3B Instruct (base) | 91.7%  | 23.0 min |
| Fine-tuned (QLoRA) | 99.3%  | 3.6 min |

 
- **Valid query** means the generated SQL runs against the table schema (checked with SQLite). 

## What the model does
 
The model receives a schema and a question, and completes the prompt after `### SQL:`.
 
```text
### Instruction:
Write the SQL query that answers the question, using the table schema.
 
### Schema:
CREATE TABLE table_name_45 (class VARCHAR, city_of_license VARCHAR)
 
### Question:
What class is the city of black mountain, north carolina?
 
### SQL:
SELECT class FROM table_name_45 WHERE city_of_license = "black mountain, north carolina"
```
 
This is a training example, shown to illustrate the format.
 
## Method
 
**Data**
- Dataset: [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) (78,577 examples).
- Cleaning: missing values and empty strings removed.
- Split: shuffled with a fixed seed, then **7,000 train / 200 validation / 300 test**.
- Longest training example: 228 tokens (mean 83), so no example is truncated at `max_seq_length = 512`.
  
**Model and training**
- Base model: `unsloth/llama-3.2-3b-instruct-bnb-4bit` (4-bit quantization).
- LoRA on all attention and MLP projection layers (`q, k, v, o, gate, up, down`), rank 32, alpha 32, no dropout: 48.6M trainable parameters out of 3.26B (1.49%).
- 2 epochs (1,750 steps), effective batch size 8 (1 × 8 gradient accumulation), learning rate 1e-4 with a cosine schedule, 8-bit AdamW, fp16.
- Validation every 100 steps; the checkpoint with the lowest validation loss is reloaded at the end.
## Evaluation
 
For each test example, the model generates a query (greedy decoding, no sampling). The query is then extracted (first `SELECT` statement) and checked with SQLite:
 
1. the table described by the schema is created in an in-memory database,
2. `EXPLAIN <query>` is run. `EXPLAIN` asks the engine how it would execute the query **without executing it**, and fails if the SQL is wrong or refers to a table or column that does not exist.
The base model is evaluated the same way by temporarily disabling the LoRA adapter. As a sanity check, the dataset's own reference queries pass this test 99.0% of the time, so the remaining failures are mostly SQLite dialect differences rather than model errors.

## Reproduce
1. Installing Unsloth (`pip install unsloth`, and installs most of its dependencies (`transformers`, `trl`, `peft`, `bitsandbytes`,`datasets`). 

2. Run all cells from top to bottom. The notebook trains the adapter, and runs the evaluation.
 
 
