export MS_ENABLE_ACLNN=1
export GRAPH_OP_RUN=1
python opensora/sample/sample_t2v.py \
    --model_path LanguageBind/Open-Sora-Plan-v1.0.0 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --text_prompt examples/prompt_list_0.txt \
    --ae CausalVAEModel_4x8x8 \
    --version 65x512x512 \
    --save_img_path "./sample_images/prompt_list_0" \
    --fps 24 \
    --guidance_scale 4.5 \
    --num_sampling_steps 250 \
    --force_images
