import numpy as np
from transformers import AutoTokenizer

import mindspore as ms
from mindspore import ops
from mindspore.dataset import GeneratorDataset

from dataset import VideoDatasetWithResizing


def collate_fn(examples):
    text_input_ids = [x["text_input_ids"] for x in examples]
    text_input_ids = np.stack(text_input_ids)

    videos = [x["video"] for x in examples]
    videos = np.stack(videos)

    return videos, text_input_ids


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "THUDM/CogVideoX-5b",
        subfolder="tokenizer",
    )

    dataset1 = VideoDatasetWithResizing(
        data_root="/home/townwish/codes/diffusers_dev/datasets/video-dataset-disney",
        caption_column="prompt.txt",
        video_column="videos.txt",
        max_num_frames=49,
        height_buckets=[480],
        width_buckets=[720],
        frame_buckets=[49],
        load_tensors=False,
        tokenizer=tokenizer,
        max_sequence_length=226,
        use_rotary_positional_embeddings=False,
        vae_scale_factor_spatial=8,
        patch_size=2,
        video_reader_backend="decord",
    )

    dataset2 = VideoDatasetWithResizing(
        data_root="/home/townwish/codes/diffusers_dev/datasets/video-dataset-disney",
        caption_column="prompt.txt",
        video_column="videos.txt",
        max_num_frames=49,
        height_buckets=[480],
        width_buckets=[720],
        frame_buckets=[49],
        load_tensors=False,
        tokenizer=tokenizer,
        max_sequence_length=226,
        use_rotary_positional_embeddings=False,
        vae_scale_factor_spatial=8,
        patch_size=2,
        video_reader_backend="cv2",
    )


    dataloader1 = GeneratorDataset(
        dataset1, column_names=["examples"]
    ).batch(
        batch_size=1,
        per_batch_map=lambda examples, batch_info: collate_fn(examples),
        input_columns=["examples"],
        output_columns=["videos", "text_input_ids"],
    ).create_tuple_iterator()

    dataloader2 = GeneratorDataset(
        dataset2, column_names=["examples"]
    ).batch(
        batch_size=1,
        per_batch_map=lambda examples, batch_info: collate_fn(examples),
        input_columns=["examples"],
        output_columns=["videos", "text_input_ids"],
    ).create_tuple_iterator()

    for idx, (data1, data2) in enumerate(zip(dataloader1, dataloader2)):
        breakpoint()
        print(
            f"iter{idx}: videos{ops.all(data1[0] == data2[0])}; text_input_ids{ops.all(data1[1] == data2[1])}"
        )
