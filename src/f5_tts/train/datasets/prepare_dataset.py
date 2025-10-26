import argparse
import json
import shutil
from pathlib import Path
import torchaudio
import torch
from datasets import load_dataset
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

from f5_tts.model.utils import convert_char_to_pinyin, convert_char_to_jyutping


PRETRAINED_VOCAB_PATH = (
    Path(__file__).parents[4] / "data" / "Emilia_ZH_EN_pinyin" / "vocab.txt"
)


def save_audio_from_dict(audio_dict, output_path):
    """Save audio from dict with 'array' and 'sampling_rate' to wav file."""

    array = (
        torch.tensor(audio_dict["array"])
        if not isinstance(audio_dict["array"], torch.Tensor)
        else audio_dict["array"]
    )
    sampling_rate = audio_dict["sampling_rate"]
    torchaudio.save(output_path, array, sampling_rate)


def prepare_hf_dataset(
    dataset_name, split, text_column, phone_column, audio_column, lang="zh"
):
    """Load and prepare dataset from HuggingFace."""
    print(f"Loading dataset {dataset_name} with split {split}...")
    dataset = load_dataset(dataset_name, split=split)

    # Create temporary directory for audio files
    temp_audio_dir = Path(f"temp_wavs_{dataset_name.replace('/', '_')}_{split}")
    temp_audio_dir.mkdir(exist_ok=True)

    processed_data = []
    durations = []
    vocab_set = set()

    polyphone = True

    print("Processing dataset...")
    for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
        # Extract columns
        text = item[text_column]
        audio_dict = item[audio_column]

        # Save audio to file
        audio_filename = f"{idx:06d}.wav"
        audio_path = temp_audio_dir / audio_filename
        save_audio_from_dict(audio_dict, str(audio_path))

        # Get duration
        duration = audio_dict["array"].shape[-1] / audio_dict["sampling_rate"]
        durations.append(duration)

        # Handle phonemes
        if phone_column and phone_column in item:
            phonemes = item[phone_column]
            processed_text = phonemes
        else:
            # Convert text to phonemes
            if lang == "zh":
                processed_text = convert_char_to_pinyin([text], polyphone=polyphone)[0]
            elif lang == "yue":
                processed_text = convert_char_to_jyutping([text])[0]
            else:
                raise ValueError(f"Unsupported language: {lang}")

        processed_data.append(
            {
                "audio_path": str(audio_path),
                "text": processed_text,
                "duration": duration,
            }
        )

        vocab_set.update(list(processed_text))

    return processed_data, durations, vocab_set, temp_audio_dir


def save_prepped_dataset(
    out_dir, result, duration_list, text_vocab_set, is_finetune, lang
):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving to {out_dir} ...")

    raw_arrow_path = out_dir / "raw.arrow"
    with ArrowWriter(path=raw_arrow_path.as_posix()) as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)
        writer.finalize()

    # Save durations to JSON
    dur_json_path = out_dir / "duration.json"
    with open(dur_json_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # Handle vocab file - write only once based on finetune flag
    voca_out_path = out_dir / "vocab.txt"
    if is_finetune:
        if lang == "zh":
            file_vocab_finetune = PRETRAINED_VOCAB_PATH.as_posix()
        elif lang == "yue":
            jyutping_vocab_path = (
                Path(__file__).parents[4] / "data" / "jyutping" / "vocab.txt"
            )
            file_vocab_finetune = jyutping_vocab_path.as_posix()
        else:
            raise ValueError(f"Unsupported language for finetune: {lang}")
        shutil.copy2(file_vocab_finetune, voca_out_path)
    else:
        with open(voca_out_path.as_posix(), "w", encoding="utf-8") as f:
            for vocab in sorted(text_vocab_set):
                f.write(vocab + "\n")

    dataset_name = out_dir.stem
    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")


def prepare_and_save_hf_dataset(
    dataset_name,
    split,
    text_column,
    phone_column,
    audio_column,
    out_dir,
    is_finetune: bool = True,
    lang: str = "zh",
):
    if is_finetune:
        assert (
            PRETRAINED_VOCAB_PATH.exists()
        ), f"pretrained vocab.txt not found: {PRETRAINED_VOCAB_PATH}"
    sub_result, durations, vocab_set, temp_audio_dir = prepare_hf_dataset(
        dataset_name, split, text_column, phone_column, audio_column, lang
    )
    save_prepped_dataset(out_dir, sub_result, durations, vocab_set, is_finetune, lang)

    # Clean up temp audio dir
    shutil.rmtree(temp_audio_dir)


def cli():
    parser = argparse.ArgumentParser(
        description="Prepare dataset from HuggingFace.",
        epilog="""
Examples:
    python prepare_dataset.py --dataset my_dataset --split train --text-column text --phone-column phone --audio-column audio --out-dir /output/path
        """,
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="HuggingFace dataset name."
    )
    parser.add_argument(
        "--split", type=str, required=True, help="Dataset split (e.g., train, test)."
    )
    parser.add_argument(
        "--text-column", type=str, default="text", help="Column name for text."
    )
    parser.add_argument(
        "--phone-column",
        type=str,
        default="phone",
        help="Column name for phonemes (optional).",
    )
    parser.add_argument(
        "--audio-column", type=str, default="audio", help="Column name for audio."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory to save the prepared data.",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Enable for new pretrain, otherwise is a fine-tune",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="zh",
        choices=["zh", "yue"],
        help="Language for vocab and text processing (zh for Mandarin, yue for Cantonese)",
    )
    args = parser.parse_args()

    prepare_and_save_hf_dataset(
        args.dataset,
        args.split,
        args.text_column,
        args.phone_column,
        args.audio_column,
        args.out_dir,
        is_finetune=not args.pretrain,
        lang=args.lang,
    )


if __name__ == "__main__":
    cli()
