import os
import torch
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate
from tqdm import tqdm

MODEL_ID = "Finnish-NLP/whisper-large-finnish-v3"

# Point this to your extracted Common Voice 23 root directory
# Example: CV23_ROOT = "/data/commonvoice/cv-corpus-23.0-2023-12-06"
CV23_ROOT = "/home/marko/Datasets/cv-corpus-23.0-2025-09-05/fi"
CV23_LOCALE = "fi"

# Datasets and text fields exactly as in the model card (plus local CV23)
DATASETS = {
    # Local Common Voice 23 (Finnish)
    "cv23": {
        "type": "common_voice_local",
        "root": CV23_ROOT,
        "locale": CV23_LOCALE,
        "split": "test",
        "text_raw": "sentence",
        "text_norm": "sentence",
    },

    # (Optional) keep FLEURS via HF
    "fleurs": {
        "type": "hf",
        "hf_id": "google/fleurs",
        "config": "fi_fi",
        "split": "test",
        "text_raw": "raw_transcription",
        "text_norm": "transcription",
    },
}

def load_model_and_processor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
		#torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()
    return model, processor, device

def _resolve_cv_locale_dir(root: str, locale: str) -> str:
    # Prefer root/locale if it exists; otherwise assume root is already the locale dir.
    candidate = os.path.join(root, locale)
    return candidate if os.path.isdir(candidate) else root

def _resolve_cv_tsv(locale_dir: str, split: str) -> str:
    # Common Voice typically has: train.tsv / dev.tsv / test.tsv / validated.tsv
    tsv = os.path.join(locale_dir, f"{split}.tsv")
    if os.path.isfile(tsv):
        return tsv

    # A couple of helpful fallbacks (in case someone wants "test" but only has dev/validated)
    if split == "test":
        for alt in ("dev.tsv", "validated.tsv"):
            alt_path = os.path.join(locale_dir, alt)
            if os.path.isfile(alt_path):
                return alt_path

    raise FileNotFoundError(f"Could not find TSV for split='{split}' under: {locale_dir}")

def load_common_voice_local(root: str, locale: str, split: str):
    """
    Loads a local Common Voice split from TSV + clips/*.mp3 using `datasets`.
    Produces a dataset with an 'audio' column compatible with `Audio(...)`.
    """
    locale_dir = _resolve_cv_locale_dir(root, locale)
    tsv_path = _resolve_cv_tsv(locale_dir, split)
    clips_dir = os.path.join(locale_dir, "clips")
    if not os.path.isdir(clips_dir):
        raise FileNotFoundError(f"Could not find clips/ directory at: {clips_dir}")

    # Load TSV (tab-separated). Common Voice TSV has a 'path' column and typically 'sentence'.
    ds = load_dataset(
        "csv",
        data_files={split: tsv_path},
        delimiter="\t",
        keep_default_na=False,
    )[split]

    # Build an 'audio' column from clips/<path>
    def add_audio_path(ex):
        rel = ex.get("path")
        if not rel:
            return {"audio": None}
        return {"audio": os.path.join(clips_dir, rel)}

    ds = ds.map(add_audio_path)

    # Drop rows that are missing audio or transcript
    ds = ds.filter(lambda x: x["audio"] is not None and x.get("sentence") not in (None, ""))

    # Decode + resample to 16kHz (Whisper default)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds

def load_dataset_from_cfg(cfg: dict):
    if cfg.get("type") == "common_voice_local":
        return load_common_voice_local(cfg["root"], cfg["locale"], cfg["split"])

    # Default: Hugging Face-hosted dataset
    return load_dataset(cfg["hf_id"], cfg["config"], split=cfg["split"]).cast_column(
        "audio", Audio(sampling_rate=16000)
    )

def evaluate_dataset(key: str, max_samples: int | None = None):
    cfg = DATASETS[key]
    print(f"\n=== Evaluating {key.upper()} ===")

    if cfg.get("type") == "common_voice_local":
        print(f"Local Common Voice: root={cfg['root']} / locale={cfg['locale']} / split={cfg['split']}")
    else:
        print(f"{cfg['hf_id']} / {cfg['config']} / split={cfg['split']}")

    # Load dataset (local or HF)
    ds = load_dataset_from_cfg(cfg)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    model, processor, device = load_model_and_processor()

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    normalizer = BasicTextNormalizer()

    preds = []
    refs_raw = []
    refs_norm = []

    for example in tqdm(ds, desc=f"Inference on {key}"):

        audio = example["audio"]["array"]

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                max_new_tokens=225,
            )

        pred = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        ref_raw = example[cfg["text_raw"]]
        ref_norm = example[cfg["text_norm"]]

        preds.append(pred)
        refs_raw.append(ref_raw)
        refs_norm.append(ref_norm)

    # 1) "Plain" metrics
    wer_plain = wer_metric.compute(predictions=preds, references=refs_raw) * 100
    cer_plain = cer_metric.compute(predictions=preds, references=refs_raw) * 100

    # 2) Normalized metrics (Whisper-style)
    preds_norm = [normalizer(p).strip() for p in preds]
    refs_norm_whisper = [normalizer(r).strip() for r in refs_norm]

	# Prints all the rows that had errors
    for x, y in zip(preds_norm, refs_norm_whisper):
        if x != y:
            print("Pred:" + x)
            print("Refn:" + y)

    wer_norm = wer_metric.compute(predictions=preds_norm, references=refs_norm_whisper) * 100
    cer_norm = cer_metric.compute(predictions=preds_norm, references=refs_norm_whisper) * 100

    print(f"\nResults on {key.upper()}:")
    print(f"  WER (raw):        {wer_plain:.2f}")
    print(f"  WER (normalized): {wer_norm:.2f}")
    print(f"  CER (raw):        {cer_plain:.2f}")
    print(f"  CER (normalized): {cer_norm:.2f}")

    return {
        "wer": wer_plain,
        "wer_normalized": wer_norm,
        "cer": cer_plain,
        "cer_normalized": cer_norm,
    }

if __name__ == "__main__":
    evaluate_dataset("cv23", max_samples=None)
    # evaluate_dataset("fleurs", max_samples=None)
