# Evaluation

## Using the CLI for Evaluation

To evaluate a model, we can simply use the Oumi CLI, e.g.:

```{code-block} bash
oumi evaluate -c configs/oumi/phi3.eval.lm_harness.yaml
```

The `-c` flag specifies the path to your evaluation configuration file. See {typer}`oumi evaluate` for more details.

To customize which model and tasks to evaluate, you can create a new configuration file. For example:

```{code-block} yaml
model:
  model_name: "microsoft/Phi-3-mini-4k-instruct"
  trust_remote_code: True

lm_harness_params:
  tasks:
    - "huggingface_leaderboard_v1"
```

```{note}
Adjust the parameters according to your specific evaluation needs. For a complete list of configuration options, refer to {doc}`Configuration Guide <../../get_started/configuration>` page and {py:class}`~oumi.core.configs.Evaluation` class.
```

## Language Model Evaluation Harness

For evaluation, we use the `lm_eval` library, which includes a wide range of tasks such as MMLU (Massive Multitask Language Understanding), HellaSwag, TruthfulQA, and more. For more information, visit the <https://github.com/EleutherAI/lm-evaluation-harness> repository.

```{tip}
To use `lm_eval` with `oumi`:
1. Specify the tasks in your configuration file under `lm_harness_params.tasks`.
2. Adjust other parameters like `num_fewshot` and `num_samples` as needed.
```

### Supported Tasks

For a full list of supported tasks, refer to the official list: <https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md>.

Locally, you can also run:

```{code-block} bash
lm-eval --tasks list
```

### Custom Evaluation Tasks

To build a custom evaluation task, refer to the {doc}`Custom Evaluation Tasks <../../advanced/custom_evaluation>`.

## Troubleshooting

If you encounter issues during evaluation, check the {doc}`troubleshooting guide <../../faq/troubleshooting>` or open an issue on the [Oumi GitHub repository](https://github.com/oumi-ai/oumi/issues).
