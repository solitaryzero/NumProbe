This repository contains data and code for paper [Language Models Encode the Value of Numbers Linearly](https://arxiv.org/abs/2401.03735).

# Content
The `src/` folder contains code used by the paper, while the `scripts/` folder contains scripts executing the experiments.

This repository uses the [LLaMA-2 family models](https://huggingface.co/meta-llama/Llama-2-7b-hf) and [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) from the huggingface model hub.

# Usage
Before executing the experiments, please download the LLMs mentioned before, and change the `model_path` variable in all `get_embeds.sh` to your path to the corresponding LLMs.

The experiments in different sections correspond to different scripts:
* Section 3 (Existence of Encoded Numbers): `scripts/*.sh`
* Section 4 (Utilization of Encoded Numbers): `scripts/intervene/*.sh`
* Appendix C (Subtraction Problems): `scripts/subtraction/*.sh`
* Appendix F (Partial Number Encoding): `scripts/partial/*.sh`
* Appendix G (Control Tasks): `scripts/control/*.sh`
* Appendix J (Adding Probed Numbers): `scripts/addition/*.sh`
* Additional experiments (Prompt Robustness): `scripts/robustness/*.sh`

## Existence of Encoded Numbers
1. `scripts/get_embeds.sh` to achieve hidden states in LLMs.
2. `scripts/refactor_embeds.sh` to refactor the outputs.
3. `scripts/probe.sh` to train linear probes, and `probe_mlp.sh` to train MLP probes.
4. `scripts/analyze.sh` to analyze the features of linear probes, and `analyze_MLP.sh` for those of MLP probes.
5. `scripts/draw.sh` to generate the figures about metric trends.
6. `scripts/draw_pattern.sh` to generate the figures about probe prediction patterns across different layers.

## Activation Patching
1. `scripts/intervene/get_embeds.sh` to achieve hidden states in LLMs.
2. `scripts/intervene/refactor_embeds.sh` to refactor the outputs.
3. `scripts/intervene/probe.sh` to train linear probes.
4. `scripts/intervene/intervene_patch.sh` to get the results of activation patching.
5. `scripts/intervene/draw_patch.sh` to generate the figures about activation patching.

## Linear Intervention
Step 1~3 are identical to those in Section "Activation Patching". Skip these steps if needed.

1. `scripts/intervene/get_embeds.sh` to achieve hidden states in LLMs.
2. `scripts/intervene/refactor_embeds.sh` to refactor the outputs.
3. `scripts/intervene/probe.sh` to train linear probes.
4. `scripts/intervene/intervene_linear.sh` to get the results of activation patching with a positive delta.
5. `scripts/intervene/intervene_linear_minus.sh` to get the results of activation patching with a negative delta.
6. `scripts/intervene/filter_linear.sh` to refactor the linear intervention results and filter out nonsense outputs.
7. `scripts/intervene/draw.sh` to generate the figures about linear intervention.
8. (Optional) For experiments in Appendix I.1, run scripts under the `scripts/intervene/optional` folder.

## Subtraction Problems
1. `scripts/subtraction/get_embeds.sh` to achieve hidden states in LLMs.
2. `scripts/subtraction/refactor_embeds.sh` to refactor the outputs.
3. `scripts/subtraction/probe.sh` to train linear probes.
4. `scripts/subtraction/analyze.sh` to analyze the features of linear probes.
5. `scripts/subtraction/draw.sh` to generate the figures about Subtraction Problems.

## Partial Number Encoding
1. `scripts/partial/get_embeds.sh` to achieve hidden states in LLMs.
2. `scripts/partial/refactor_embeds.sh` to refactor the outputs.
3. `scripts/partial/probe.sh` to train linear probes.
4. `scripts/partial/analyze.sh` to analyze the features of linear probes.
5. `scripts/partial/draw.sh` to generate the figures about Partial Number Encoding.

## Control Tasks
The embeddings used for this experiment is generated in Section "Existence of Encoded Numbers".

3. `scripts/control/probe.sh` to train linear probes.
4. `scripts/control/analyze.sh` to analyze the features of linear probes.
5. `scripts/control/draw.sh` to generate the figures about Control Tasks.

## Adding Probed Numbers
1. `scripts/addition/get_embeds.sh` to achieve hidden states in LLMs.
2. `scripts/addition/refactor_embeds.sh` to refactor the outputs.
3. `scripts/addition/probe.sh` to train linear probes.
4. `scripts/addition/analyze.sh` to analyze the features of linear probes.
5. `scripts/addition/draw.sh` to generate the figures about Adding Probed Numbers.

## Prompt Robustness (Optional)
The probes used for this experiment is trained in Section "Existence of Encoded Numbers".

1. `scripts/robustness/get_embeds.sh` to achieve hidden states in LLMs.
2. `scripts/robustness/refactor_embeds.sh` to refactor the outputs.
3. `scripts/robustness/analyze.sh` to analyze the features of linear probes.
4. `scripts/robustness/draw.sh` to generate the figures about Prompt Robustness.