import os
import pathlib
import random
import string
import tempfile
from typing import Iterable, List

import spaces
import gradio as gr
import huggingface_hub
import torch
import yaml
from gradio_logsview.logsview import Log, LogsView, LogsViewRunner
from mergekit.config import MergeConfiguration

from clean_community_org import garbage_collect_empty_models
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import timezone

MARKDOWN_DESCRIPTION = """
# mergekit-gui

The fastest way to perform a model merge 🔥

Specify a YAML configuration file (see examples below) and a HF token and this app will perform the merge and upload the merged model to your user profile. Uses Zero GPU quota to perform the merge.
"""

MARKDOWN_ARTICLE = """
___

## Merge Configuration

[Mergekit](https://github.com/arcee-ai/mergekit) configurations are YAML documents specifying the operations to perform in order to produce your merged model.
Below are the primary elements of a configuration file:

- `merge_method`: Specifies the method to use for merging models. See [Merge Methods](https://github.com/arcee-ai/mergekit#merge-methods) for a list.
- `slices`: Defines slices of layers from different models to be used. This field is mutually exclusive with `models`.
- `models`: Defines entire models to be used for merging. This field is mutually exclusive with `slices`.
- `base_model`: Specifies the base model used in some merging methods.
- `parameters`: Holds various parameters such as weights and densities, which can also be specified at different levels of the configuration.
- `dtype`: Specifies the data type used for the merging operation.
- `tokenizer_source`: Determines how to construct a tokenizer for the merged model.

## Merge Methods

A quick overview of the currently supported merge methods:

| Method                                                                                       | `merge_method` value | Multi-Model | Uses base model |
| -------------------------------------------------------------------------------------------- | -------------------- | ----------- | --------------- |
| Linear ([Model Soups](https://arxiv.org/abs/2203.05482))                                     | `linear`             | ✅          | ❌              |
| SLERP                                                                                        | `slerp`              | ❌          | ✅              |
| [Task Arithmetic](https://arxiv.org/abs/2212.04089)                                          | `task_arithmetic`    | ✅          | ✅              |
| [TIES](https://arxiv.org/abs/2306.01708)                                                     | `ties`               | ✅          | ✅              |
| [DARE](https://arxiv.org/abs/2311.03099) [TIES](https://arxiv.org/abs/2306.01708)            | `dare_ties`          | ✅          | ✅              |
| [DARE](https://arxiv.org/abs/2311.03099) [Task Arithmetic](https://arxiv.org/abs/2212.04089) | `dare_linear`        | ✅          | ✅              |
| Passthrough                                                                                  | `passthrough`        | ❌          | ❌              |
| [Model Stock](https://arxiv.org/abs/2403.19522)                                              | `model_stock`        | ✅          | ✅              |


## Citation

This GUI is powered by [Arcee's MergeKit](https://arxiv.org/abs/2403.13257).
If you use it in your research, please cite the following paper:

```
@article{goddard2024arcee,
  title={Arcee's MergeKit: A Toolkit for Merging Large Language Models},
  author={Goddard, Charles and Siriwardhana, Shamane and Ehghaghi, Malikeh and Meyers, Luke and Karpukhin, Vlad and Benedict, Brian and McQuade, Mark and Solawetz, Jacob},
  journal={arXiv preprint arXiv:2403.13257},
  year={2024}
}
```

This Space is heavily inspired by LazyMergeKit by Maxime Labonne (see [Colab](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb)).
"""

examples = [[str(f)] for f in pathlib.Path("examples").glob("*.yaml")]

# Do not set community token as `HF_TOKEN` to avoid accidentally using it in merge scripts.
# `COMMUNITY_HF_TOKEN` is used to upload models to the community organization (https://huggingface.co/mergekit-community)
# when user do not provide a token.
COMMUNITY_HF_TOKEN = os.getenv("COMMUNITY_HF_TOKEN")


def run_merge_cpu(runner: LogsViewRunner, cli: str, merged_path: str, tmpdirname: str):
    # Set tmp HF_HOME to avoid filling up disk Space
    tmp_env = os.environ.copy()
    tmp_env["HF_HOME"] = f"{tmpdirname}/.cache"
    full_cli = cli + f" --lora-merge-cache {tmpdirname}/.lora_cache --transformers-cache {tmpdirname}/.cache"
    yield from runner.run_command(full_cli.split(), cwd=merged_path, env=tmp_env)
    yield ("done", runner.exit_code)


@spaces.GPU(duration=60 * 5)
def run_merge_gpu(runner: LogsViewRunner, cli: str, merged_path: str, tmpdirname: str):
    yield from run_merge_cpu(
        runner,
        cli + " --cuda --low-cpu-memory --read-to-gpu",
        merged_path,
        tmpdirname,
    )


def run_merge(
    runner: LogsViewRunner,
    cli: str,
    merged_path: str,
    tmpdirname: str,
    use_gpu: bool,
):
    if use_gpu:
        yield from run_merge_gpu(runner, cli, merged_path, tmpdirname)
    else:
        yield from run_merge_cpu(runner, cli, merged_path, tmpdirname)


def prefetch_models(
    runner: LogsViewRunner,
    merge_config: MergeConfiguration,
    hf_home: str,
    lora_merge_cache: str,
):
    for model in merge_config.referenced_models():
        yield runner.log(f"Downloading {model}...")
        model = model.merged(cache_dir=lora_merge_cache, trust_remote_code=False)
        local_path = model.local_path(cache_dir=hf_home)
        yield runner.log(f"\tDownloaded {model} to {local_path}")


def merge(yaml_config: str, hf_token: str, repo_name: str, private: bool) -> Iterable[List[Log]]:
    runner = LogsViewRunner()

    if not yaml_config:
        yield runner.log("Empty yaml, enter your config or pick an example below", level="ERROR")
        return
    try:
        merge_config = MergeConfiguration.model_validate(yaml.safe_load(yaml_config))
    except Exception as e:
        yield runner.log(f"Invalid yaml {e}", level="ERROR")
        return

    is_community_model = False
    if not hf_token:
        if private:
            yield runner.log(
                "Cannot upload model as private without a token. Please provide a HF token.",
                level="ERROR",
            )
            return
        if "/" in repo_name and not repo_name.startswith("mergekit-community/"):
            yield runner.log(
                f"Cannot upload merge model to namespace {repo_name.split('/')[0]}: you must provide a valid token.",
                level="ERROR",
            )
            return
        yield runner.log(
            "No HF token provided. Your merged model will be uploaded to the https://huggingface.co/mergekit-community organization."
        )
        is_community_model = True
        if not COMMUNITY_HF_TOKEN:
            raise gr.Error("Cannot upload to community org: community token not set by Space owner.")
        hf_token = COMMUNITY_HF_TOKEN

    api = huggingface_hub.HfApi(token=hf_token)
    has_gpu = torch.cuda.is_available()
    cli = "mergekit-yaml config.yaml merge --copy-tokenizer --allow-crimes -v" + (
        " --out-shard-size 1B --lazy-unpickle" if (not has_gpu) else ""
    )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
        tmpdir = pathlib.Path(tmpdirname)
        merged_path = tmpdir / "merged"
        merged_path.mkdir(parents=True, exist_ok=True)
        config_path = merged_path / "config.yaml"
        config_path.write_text(yaml_config)
        yield runner.log(f"Merge configuration saved in {config_path}")

        if not repo_name:
            yield runner.log("No repo name provided. Generating a random one.")
            repo_name = f"mergekit-{merge_config.merge_method}"
            # Make repo_name "unique" (no need to be extra careful on uniqueness)
            repo_name += "-" + "".join(random.choices(string.ascii_lowercase, k=7))
            repo_name = repo_name.replace("/", "-").strip("-")

        if is_community_model and not repo_name.startswith("mergekit-community/"):
            repo_name = f"mergekit-community/{repo_name}"

        try:
            yield runner.log(f"Creating repo {repo_name}")
            repo_url = api.create_repo(repo_name, exist_ok=True, private=private, repo_type="model")
            yield runner.log(f"Repo created: {repo_url}")
        except Exception as e:
            yield runner.log(f"Error creating repo {e}", level="ERROR")
            return

        # Prefetch models to avoid downloading them with scarce GPU time
        yield runner.log("Prefetching models...")
        yield from prefetch_models(
            runner,
            merge_config,
            hf_home=f"{tmpdirname}/.cache",
            lora_merge_cache=f"{tmpdirname}/.lora_cache",
        )
        yield runner.log("Models prefetched. Starting merge.")
        exit_code = None
        try:
            for ev in run_merge(
                runner,
                cli,
                merged_path,
                tmpdirname,
                use_gpu=has_gpu,
            ):
                if isinstance(ev, tuple) and ev[0] == "done":
                    exit_code = ev[1]
                    continue
                yield ev
        except Exception as e:
            yield runner.log(f"Error running merge {e}", level="ERROR")
            yield runner.log("Merge failed. Deleting repo as no model is uploaded.", level="ERROR")
            api.delete_repo(repo_url.repo_id)
            return

        if exit_code != 0:
            yield runner.log(f"Exit code: {exit_code}")
            yield runner.log("Merge failed. Deleting repo as no model is uploaded.", level="ERROR")
            api.delete_repo(repo_url.repo_id)
            return

        yield runner.log("Model merged successfully. Uploading to HF.")
        yield from runner.run_python(
            api.upload_folder,
            repo_id=repo_url.repo_id,
            folder_path=merged_path / "merge",
        )
        yield runner.log(f"Model successfully uploaded to HF: {repo_url.repo_id}")


merge.zerogpu = True
run_merge.zerogpu = True


# This is workaround. As the space always getting stuck.
def _restart_space():
    huggingface_hub.HfApi().restart_space(
        repo_id="arcee-ai/mergekit-gui", token=COMMUNITY_HF_TOKEN, factory_reboot=False
    )


# Run garbage collection every hour to keep the community org clean.
# Empty models might exists if the merge fails abruptly (e.g. if user leaves the Space).
def _garbage_remover():
    try:
        garbage_collect_empty_models(token=COMMUNITY_HF_TOKEN)
    except Exception as e:
        print("Error running garbage collection", e)


scheduler = BackgroundScheduler()
restart_space_job = scheduler.add_job(_restart_space, "interval", seconds=21600)
garbage_remover_job = scheduler.add_job(_garbage_remover, "interval", seconds=3600)
scheduler.start()
next_run_time_utc = restart_space_job.next_run_time.astimezone(timezone.utc)

NEXT_RESTART = f"Next Restart: {next_run_time_utc.strftime('%Y-%m-%d %H:%M:%S')} (UTC)"

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN_DESCRIPTION)
    gr.Markdown(NEXT_RESTART)

    with gr.Row():
        filename = gr.Textbox(visible=False, label="filename")
        config = gr.Code(language="yaml", lines=10, label="config.yaml")
        with gr.Column():
            token = gr.Textbox(
                lines=1,
                label="HF Write Token",
                info="https://hf.co/settings/token",
                type="password",
                placeholder="Optional. Will upload merged model to MergeKit Community if empty.",
            )
            repo_name = gr.Textbox(
                lines=1,
                label="Repo name",
                placeholder="Optional. Will create a random name if empty.",
            )
            private = gr.Checkbox(
                label="Private",
                value=False,
                info="Upload the model as private. If not checked, will be public. Must provide a token.",
            )
    button = gr.Button("Merge", variant="primary")
    logs = LogsView(label="Terminal output")
    gr.Examples(
        examples,
        fn=lambda s: (s,),
        run_on_click=True,
        label="Examples",
        inputs=[filename],
        outputs=[config],
    )
    gr.Markdown(MARKDOWN_ARTICLE)

    button.click(fn=merge, inputs=[config, token, repo_name, private], outputs=[logs])


demo.queue(default_concurrency_limit=1).launch(mcp_server=True, share=True)

