if __name__ == "__main__":
    config = {
        "vocab_size": 1000,
        "dim": 512,
        "n_heads": 8,
        "n_layers": 6,
        "max_seq_len": 256,
        "dropout": 0.1,
        "batch_size": 16,
        "learning_rate": 3e-4
    }

    save_config(config, "config.json")
    cfg = load_config("config.json")

    print(cfg.vocab_size)   # 1000
    print(cfg.n_layers)     # 6

