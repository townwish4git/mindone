from mindspore import load_checkpoint, save_checkpoint


def except_vae(ckpt_file):
    sd = load_checkpoint(ckpt_file)
    filtered_sd = {key: value for key, value in sd.items() if 'first_stage_model' not in key}
    return filtered_sd


def only_vae(ckpt_file):
    sd = load_checkpoint(ckpt_file)
    filtered_sd = {key: value for key, value in sd.items() if 'first_stage_model' in key}
    return filtered_sd


def merge(vae_file, other_file):
    ret = only_vae(vae_file)
    ret.update(except_vae(other_file))
    return ret


if __name__ == "__main__":
    vae_file = ""
    other_file = ""
    save_path = ""
    sd = merge(vae_file, other_file)
    save_checkpoint(sd, save_path)
