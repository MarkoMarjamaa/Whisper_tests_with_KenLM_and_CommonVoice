import os
import torch
from datasets import load_dataset, Audio
from transformers import pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import whisper_lm_transformers  # noqa: F401  # needed to register "whisper-with-lm"
import evaluate
from tqdm import tqdm

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

# Fine-tuned Finnish Whisper
MODEL_ID = "Finnish-NLP/whisper-large-finnish-v3"

# Path to your Finnish KenLM binary
LM_PATH = "/home/marko/models/fi-5grams/kenlm_finnish.bin"  

# Whisper-LM LM parameters (you should tune these on a dev set)
LM_ALPHA = 0.39103406386645967
LM_BETA = 1.0658034417846092

# Limit evaluation for quick tests, or set to None for full test set
MAX_SAMPLES = None  # e.g. 500 for a quick run

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

# ------------------------------------------------------------------
# BUILD PIPELINE
# ------------------------------------------------------------------

def build_asr_pipeline():
	"""
	Build a Whisper-with-LM pipeline:
	  - Whisper model: Finnish-NLP/whisper-large-finnish-v3
	  - KenLM model: LM_PATH
	"""
	device = 0 if torch.cuda.is_available() else -1
	asr_pipe = pipeline(
		"whisper-with-lm",		  # registered by whisper_lm_transformers
		model=MODEL_ID,
		lm_model=LM_PATH,		   # KenLM binary path
		lm_alpha=LM_ALPHA,
		lm_beta=LM_BETA,
		language="fi",			  # force Finnish decoding
		torch_dtype=torch.float32,
		device=device,
	)
	return asr_pipe

# ------------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------------
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



def main():
	#print(f"Loading dataset: {DATASET_ID} / {DATASET_CONFIG} / {DATASET_SPLIT}")
	cfg = DATASETS["cv23"]
	if cfg.get("type") == "common_voice_local":
		print(f"Local Common Voice: root={cfg['root']} / locale={cfg['locale']} / split={cfg['split']}")
	else:
		print(f"{cfg['hf_id']} / {cfg['config']} / split={cfg['split']}")

	# Load dataset (local or HF)
	ds = load_dataset_from_cfg(cfg)

	# Resample audio to 16 kHz (Whisper default)
	ds = ds.cast_column("audio", Audio(sampling_rate=16000))

	if MAX_SAMPLES is not None:
		ds = ds.select(range(min(MAX_SAMPLES, len(ds))))

	print(f"Number of examples: {len(ds)}")

	print("Building Whisper+KenLM pipeline...")
	asr_pipe = build_asr_pipeline()

	wer_metric = evaluate.load("wer")
	cer_metric = evaluate.load("cer")
	normalizer = BasicTextNormalizer()

	preds = []
	refs_raw = []
	refs_norm = []

	for example in tqdm(ds, desc="Transcribing with Whisper+KenLM"):
		
		audio = example["audio"]["array"]
		# Whisper-with-LM pipeline; returns dict {"text": "...", ...}
		result = asr_pipe(audio, return_timestamps=True)
		pred = result["text"]

		ref_raw = example[cfg["text_raw"]]
		ref_norm = example[cfg["text_norm"]]

		preds.append(pred)
		refs_raw.append(ref_raw)
		refs_norm.append(ref_norm)

	# 1) Raw metrics (no extra normalization)
	wer_raw = wer_metric.compute(predictions=preds, references=refs_raw) * 100
	cer_raw = cer_metric.compute(predictions=preds, references=refs_raw) * 100

	# 2) Whisper-style normalized metrics
	preds_norm = [normalizer(p).strip() for p in preds]
	refs_norm_whisper = [normalizer(r).strip() for r in refs_norm]

	for x, y in zip(preds_norm, refs_norm_whisper):
		if x != y :
			print("Pred:" + x)
			print("Refn:" + y)

	wer_norm = wer_metric.compute(
		predictions=preds_norm, references=refs_norm_whisper
	) * 100
	cer_norm = cer_metric.compute(
		predictions=preds_norm, references=refs_norm_whisper
	) * 100

	print("\n=== Results: CV23 fi_fi / test with KenLM ===")
	print(f"WER (raw):		{wer_raw:.2f}")
	print(f"WER (normalized): {wer_norm:.2f}")
	print(f"CER (raw):		{cer_raw:.2f}")
	print(f"CER (normalized): {cer_norm:.2f}")

if __name__ == "__main__":
	main()

