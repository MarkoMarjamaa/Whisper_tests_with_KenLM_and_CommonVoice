# Whisper_tests_with_KenLM_and_CommonVoice
Running WER/CER test with Finnish Common Voice and Whisper with KenLM

These scripts were tested with AMD Ryzen 395+

# Install packages 
```
pip install triton-3.2.0+rocm7.1.0.git20943800-cp312-cp312-linux_x86_64.whl
pip install torch-2.6.0+rocm7.1.0.lw.git78f6ff78-cp312-cp312-linux_x86_64.whl
pip install torchaudio-2.6.0+rocm7.1.0.gitd8831425-cp312-cp312-linux_x86_64.whl
pip install torchvision-0.21.0+rocm7.1.0.git4040d51f-cp312-cp312-linux_x86_64.whl

pip install "datasets<4.0.0" "huggingface_hub<0.25.0"
pip install "transformers>=4.39" evaluate torch torchaudio
pip install soundfile
pip install jiwer
pip install librosa

pip install whisper-lm-transformers
```
# Dataset Common Voice 
These ware tested with CV23 Finnish. Remove all the characters "“” manually. Otherwise quotes don't end properly and can contain several rows. If running only test-split, fix only test.tsv.

With Amd's version of "cuda" :
  torch_dtype=torch.float32

# Scripts
```
cv_finnish_whisper_v3.py
```
Runs Common Voice 23 tests with normal Whisper. 

```
cv_whisper_lm_optimizer.py
```
Fork from whisper_lm_optimizer_with_hf.py, works with local Common Voice. Optimizes alpha/beta for KenLM. 
run like: 
```
python cv_whisper_lm_optimizer.py "Finnish-NLP/whisper-large-finnish-v3" --study_name "fi_whisper_large_v3_kenlm_opt" --dataset "/home/marko/Datasets/cv-corpus-23.0-2025-09-05/fi" --dataset_name "default" --dataset_split "test" --dataset_shuffle True --dataset_n 500 --language "fi" --beam_size 5 --lm_path "/home/marko/models/fi-5grams/kenlm_finnish.bin" --lm_alpha_min 0.0 --lm_alpha_max 1.5 --lm_beta_min 0.0 --lm_beta_max 3.0 --n_trials 50 --n_jobs 1 --n_gpus 1
```
```
cv_finnish_whisper_v3_with_kenlm.py
```
Runs Common Voice 23 tests with Whisper with KenLM. 

# Results with Finnish CV23
With Finnish-NLP/whisper-large-finnish-v3
```
WER (raw): 9.70
WER (normalized): 6.97
CER (raw): 1.64
CER (normalized): 1.24 
```
With Finnish-NLP/whisper-large-finnish-v3 & Finnish KenLM 
https://huggingface.co/Finnish-NLP/wav2vec2-xlsr-1b-finnish-lm-v2/tree/main/language_model
```
WER (raw): 6.42
WER (normalized): 3.34
CER (raw): 1.13
CER (normalized): 0.69 
```
