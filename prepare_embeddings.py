from transformers import TransfoXLTokenizer, TransfoXLModel
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer
import numpy as np
import torch
import os
import tqdm
from variables import avalon_templates, dmlab_templates


def get_vocab(tokenizer, vocab_size):
    vocab = []
    if isinstance(tokenizer, SimpleTokenizer):
        for i in range(vocab_size):
            vocab.append(tokenizer.decode([i]))
        vocab = np.array(vocab)
    else:
        vocab = np.array(tokenizer.convert_ids_to_tokens(np.arange(vocab_size)))
    return vocab


def get_vocab_intersection(src_vocab, tar_vocab):
    targets = []
    srcs = []
    for i, s in tqdm.tqdm(enumerate(src_vocab), desc="Creating vocabulary overlap..."):
        occ = (s == tar_vocab)
        if occ.any():
            tar_idx = np.nonzero(occ)
            if len(tar_idx) == 1:
                targets.extend(tar_idx[0])
                srcs.append(i)

    return np.array(srcs), np.array(targets)


def get_clip_embs(tokenized, clip_net, device='cuda', batch_size=128):
    clip_embs = []
    for i in range(0, len(tokenized), batch_size):
        with torch.no_grad():
            tok_emb = clip_net.encode_text(tokenized[i:i+batch_size].to(device))
            clip_embs.append(tok_emb.float().cpu().numpy())
    clip_embs = np.concatenate(clip_embs)
    return clip_embs


def main():
    if not os.path.exists('data'):
        os.makedirs('data', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = "ViT-B/16"

    model, preprocess = clip.load(encoder)
    encoder = encoder.replace("/", "")

    model.cuda().eval()
    clip_vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Vocab size:", clip_vocab_size)

    clip_tokenizer = SimpleTokenizer()

    if not os.path.exists('data/clip_vocab.npy'):
        print("Dumping vocab...")
        clip_vocab = get_vocab(clip_tokenizer, clip_vocab_size)
        np.save('data/clip_vocab', clip_vocab)
    else:
        clip_vocab = np.load('data/clip_vocab.npy')

    if not os.path.exists(f'data/{encoder}_embs.npz'):
        print("Dumping Embeddings...")
        # dump all clip embeddings
        clip_tokenized = torch.stack([clip.tokenize([tok]) for tok in clip_vocab])
        clip_embs = get_clip_embs(clip_tokenized.squeeze(), model, device)
        # np.save(open(f'data/{encoder}_{lm}_vocab.npz', 'wb'), clip_vocab)
        np.save(open(f'data/{encoder}_embs.npz', 'wb'), clip_embs.squeeze())

    tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')

    if not os.path.exists(f'data/transfo-xl-wt103_clip_intersect.npz'):
        print("Dumping vocabulary overlap...")
        transformer_vocab = get_vocab(tokenizer, tokenizer.vocab_size)
        vocab_intersect_clip, vocab_intersect_lm = get_vocab_intersection(clip_vocab, transformer_vocab)
        np.save(open(f'data/clip_transfo-xl-wt103_intersect.npz', 'wb'), vocab_intersect_clip)

    if not os.path.exists(f'data/{encoder}_dmlab_prompt_embs.npz'):
        print("Dumping dmlab prompt embeddings...")
        clip_embs = []
        for tok in tqdm.tqdm(clip_vocab):
            prompted = [p.format(tok) for p in dmlab_templates]
            tokenized = clip.tokenize(prompted).to(device)
            with torch.no_grad():
                vecs = model.encode_text(tokenized).cpu().mean(0).numpy()
            clip_embs.append(vecs)
        clip_embs = np.array(clip_embs)
        np.save(open(f'data/{encoder}_dmlab_prompt_embs.npz', 'wb'), clip_embs.squeeze())

    if not os.path.exists(f'data/{encoder}_avalon_prompt_embs.npz'):
        print("Dumping avalon prompt embeddings...")
        clip_embs = []
        for tok in tqdm.tqdm(clip_vocab):
            prompted = [p.format(tok) for p in avalon_templates]
            tokenized = clip.tokenize(prompted).to(device)
            with torch.no_grad():
                vecs = model.encode_text(tokenized).cpu().mean(0).numpy()
            clip_embs.append(vecs)
        clip_embs = np.array(clip_embs)
        np.save(open(f'data/{encoder}_avalon_prompt_embs.npz', 'wb'), clip_embs.squeeze())


if __name__ == '__main__':
    main()