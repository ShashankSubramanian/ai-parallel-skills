# Parallelize this ViT training case
/ai-parallelism-master 

The current directory is the target workspace. Read `vit.py`, `config.yaml`,
and `train.py` in full. Do not modify them. I want to parallelize the ViT model. 
Create all outputs under `distributed/`.

Target hardware:

- 32 NVIDIA A100 40GB GPUs.
- 8 nodes with 4 NVLink-connected GPUs per node.
- Also can support all the way to 128 GPUs where the selected mesh is valid.
- Preserve the configured global batch size and prioritize training throughput.

Complete the ai-parallelism-master workflow and create the required test and training
launch scripts. Do not execute those scripts; the user will run them on the
target hardware. Document the exact commands and mark their results unverified.

Do not inspect another evaluation run or any existing implementation of this
case.
