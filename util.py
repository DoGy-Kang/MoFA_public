import argparse


def getParser():
    """Command-line arguments for `run.py`."""
    parser = argparse.ArgumentParser(
        description="Train a softmax-only baseline or a MoFA GPT-2 model on OpenWebText."
    )
    parser.add_argument("-seed", "--seed", type=int, required=True, help="Random seed")
    parser.add_argument("-d", "--device_id", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "-m", "--mode", choices=["v", "h"], required=True,
        help="'v': softmax-only baseline, 'h': MoFA (mixed softmax/sigmoid heads)",
    )
    parser.add_argument("-n_sig", "--n_sig_head", type=int, default=0,
                        help="Number of sigmoid heads (out of 12); used with -m h")
    parser.add_argument("-n_tanh", "--n_tanh_head", type=int, default=0,
                        help="Number of tanh heads (experimental, 0 in the paper)")
    parser.add_argument("--total_steps", type=int, default=None,
                        help="Override total training steps (default: config value)")
    return parser.parse_args()
