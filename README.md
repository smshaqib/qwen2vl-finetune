# Qwen2-VL QLoRA fine-tuning — local code, Kaggle GPU

Fine-tune `Qwen2-VL-2B-Instruct` with QLoRA. Code lives in this repo; the heavy
training runs on Kaggle's free GPU via a clone-and-run notebook.

## Layout
```
.
├── train.py                  # QLoRA training loop (resumable)
├── dataset.py                # HF dataset adapter + Qwen2-VL collator
├── config.yaml               # all knobs: model, lora, data, training
├── requirements.txt
└── kaggle/run_notebook.ipynb # clones this repo + runs train.py on GPU
```

## Workflow (GitHub → Kaggle)
1. Push this repo to GitHub.
2. On https://www.kaggle.com/code → **New Notebook** → **File → Import** `kaggle/run_notebook.ipynb`
   (or copy the cells in).
3. Notebook **Settings**: Accelerator = **GPU T4 x2** (or P100), Internet = **On**.
4. Edit the `REPO` URL in cell 2, then **Run All**.
5. Trained LoRA adapter saves to `/kaggle/working/qwen2vl-lora` — download it from the
   notebook's **Output** tab, or commit the notebook to snapshot outputs.

### Iterating
Change code locally → `git push` → in the Kaggle notebook re-run cell 2 (`git pull`)
and cell 5. If a session times out, just re-run cell 5 — it **auto-resumes** from the
latest checkpoint.

## Kaggle limits to remember
- ~30 GPU hrs/week; sessions stop after ~9h or on inactivity → that's why we checkpoint often.
- T4 = 16 GB, no bf16 → config uses 4-bit + fp16.
- Scale up to `Qwen2-VL-7B-Instruct` by editing `model.name` in `config.yaml`
  (keep 4-bit, batch size 1, raise grad_accum).

## Swapping datasets
Point `data.dataset_name` / `data.split` at any HF dataset and set the three
field mappings (`image_field`, `question_field`, `answer_field`) in `config.yaml`.
No code changes needed. For captioning sets, map `answer_field` to the caption column
and set `question_field` to a column holding a fixed prompt (or adapt `build_messages`).

## Local use
You can't train here without a GPU, but you can lint / dry-run logic. The intended
GPU runtime is Kaggle.
```
pip install -r requirements.txt
python train.py --config config.yaml   # needs a CUDA GPU
```
