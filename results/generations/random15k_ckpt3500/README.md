# Random-15K Full-SFT Evaluation Outputs



This directory contains the raw evaluation logs used to compute the reported benchmark results for:



**Qwen2.5-3B-Instruct — Random-15K Full-SFT&#x20;**



The Random-15K training job was originally configured for 2 epochs, but the 24-hour ARC wall-time limit terminated the job after step 3,500. Therefore, these outputs correspond to the latest completed checkpoint of the planned optimization schedule (\~1.87/2 epochs).



## Evaluation settings



- Temperature: `0.6`

- Top-p: `0.95`

- Samples per question: `8`

- Maximum generation tokens: `32,768`

- Evaluation seed: `0`

- Prompt format: `qwen-instruct`

- Inference engine: `vLLM`

- vLLM CPU swap space: `60 GB`



## Directory structure



```text

random15k_ckpt3500/

├── aime/

│   └── AIME 2024 chunked evaluation logs

├── cn_math/

│   └── CN Math 2024 chunked evaluation logs

├── kaoyan/

│   └── Kaoyan chunked evaluation logs

├── amc/

│   └── AMC 2023 complete evaluation log

└── README.md

```



AIME and CN Math were evaluated in chunks of 5 questions. Kaoyan was evaluated in chunks of 10 questions, with the final chunk covering the remaining questions. Chunking was used to reduce memory pressure while retaining the same sampling configuration.



## Reported results



| Benchmark    | Problems | Pass\@1 | Pass\@8 |

| ------------ | -------: | ------: | ------: |

| AMC 2023     |       40 |  36.25% |  72.50% |

| AIME 2024    |       30 |   7.08% |  23.33% |

| CN Math 2024 |       30 |  13.33% |  46.67% |

| Kaoyan       |      199 |  25.69% |  65.33% |



`Pass@1` represents average correctness over the sampled generations, while `Pass@8` represents the fraction of questions for which at least one of the eight generated responses was correct.



The raw logs are retained to make the reported evaluation results auditable and reproducible.

