

from chatbot.word2sequence import Word2Sequence
import config
from tqdm import tqdm
import pickle

def save_sort_ws():
    ws = Word2Sequence()
    with open(config.sort_q_cut_path, encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            line = line.strip().split()
            ws.fit(line)

    ws.build_vocab(min_count=5)
    pickle.dump(ws, open(config.sort_ws_save_path, "wb"))