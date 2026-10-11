import json
import array, heapq, time
from collections import Counter, defaultdict

BPE_MIN_WORD_FREQ = 1      # 1 = learn merges from every word type (exactly the Assignment-1 algorithm)
_END = "</w>"


class BPETokenizer:
    """Byte-level BPE (Assignment 1). Initial symbols are the UTF-8 bytes of each word, held as chr(b), plus
    '</w>' marking the end of a word so merges never cross word boundaries. Ties in pair frequency break
    lexicographically. Text is NOT lowercased."""

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.merges = []
        self.itos = ["<pad>", "<unk>", "<s>", "</s>"]
        self.stoi = {}
        self._ranks = {}
        self._id_cache = {}

    # ---- helpers ----
    def _word_symbols(self, word):
        syms = [chr(b) for b in word.encode("utf-8")]
        syms.append(_END)
        return syms

    def _merge(self, symbols, pair):
        res, i = [], 0
        a, b = pair
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                res.append(a + b)
                i += 2
            else:
                res.append(symbols[i])
                i += 1
        return res

    # ---- training ----
    def fit_file(self, path, block_chars=50_000_000, min_word_freq=1):
        """Count word frequencies by streaming the corpus file in blocks, then learn the merges."""
        t0 = time.time()
        word_freqs = Counter()
        leftover = ""
        with open(path, "r", encoding="utf-8") as f:
            while True:
                block = f.read(block_chars)
                if not block:
                    break
                block = leftover + block
                parts = block.split()
                if parts and not block[-1].isspace():
                    leftover = parts.pop()          # the last word may be cut in half: finish it with the next block
                else:
                    leftover = ""
                word_freqs.update(parts)
        if leftover:
            word_freqs[leftover] += 1
        print(f"  counted {sum(word_freqs.values()):,} words ({len(word_freqs):,} distinct) in {time.time() - t0:.0f}s", flush=True)
        return self.fit_word_freqs(word_freqs, min_word_freq)

    def fit_word_freqs(self, word_freqs, min_word_freq=1):
        t0 = time.time()
        target = self.vocab_size
        words = [w for w, c in word_freqs.items() if c >= min_word_freq]
        freqs = [word_freqs[w] for w in words]
        splits = [self._word_symbols(w) for w in words]
        del word_freqs

        base = set()
        for syms in splits:
            base.update(syms)
        self.itos = ["<pad>", "<unk>", "<s>", "</s>"]
        self.merges = []
        self.itos.extend(sorted(base))
        print(f"  base byte alphabet: {len(base)} symbols; learning merges from {len(words):,} word types", flush=True)

        # pair_counts: how often each adjacent pair occurs (weighted by word frequency).
        # pair_words: which word types contain each pair, so a merge only revisits those words.
        pair_counts = {}
        pair_words = defaultdict(set)
        for i, syms in enumerate(splits):
            f = freqs[i]
            for j in range(len(syms) - 1):
                p = (syms[j], syms[j + 1])
                pair_counts[p] = pair_counts.get(p, 0) + f
                pair_words[p].add(i)

        # Max-heap of (-count, pair): pops the most frequent pair, ties -> lexicographically smallest pair
        # (identical to Assignment 1's min over (-count, pair)). Entries go stale when a count changes; a
        # fresh entry is pushed for every changed pair and stale ones are skipped when they reach the top.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        while len(self.itos) < target:
            best, best_f = None, 0
            while heap:
                neg_c, p = heap[0]
                if pair_counts.get(p, 0) == -neg_c:
                    best, best_f = p, -neg_c
                    heapq.heappop(heap)
                    break
                heapq.heappop(heap)                  # stale entry
            if best is None:
                break
            if best_f < 2:
                print("  stopping: no pair occurs more than once", flush=True)
                break

            self.merges.append([best[0], best[1]])
            self.itos.append(best[0] + best[1])

            changed = set()
            for i in list(pair_words[best]):
                syms, f = splits[i], freqs[i]
                for j in range(len(syms) - 1):
                    p = (syms[j], syms[j + 1])
                    c = pair_counts.get(p, 0) - f
                    if c <= 0:
                        pair_counts.pop(p, None)
                    else:
                        pair_counts[p] = c
                    pair_words[p].discard(i)
                    changed.add(p)
                new_syms = self._merge(syms, best)
                splits[i] = new_syms
                for j in range(len(new_syms) - 1):
                    p = (new_syms[j], new_syms[j + 1])
                    pair_counts[p] = pair_counts.get(p, 0) + f
                    pair_words[p].add(i)
                    changed.add(p)
            for p in changed:
                c = pair_counts.get(p)
                if c:
                    heapq.heappush(heap, (-c, p))
            del pair_words[best]

            if len(self.merges) % 1000 == 0:
                print(f"  merges {len(self.merges):,} | vocab {len(self.itos):,} | {time.time() - t0:.0f}s", flush=True)

        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self._ranks = {(a, b): r for r, (a, b) in enumerate(self.merges)}
        self._id_cache = {}
        self.vocab_size = len(self.itos)      # the size the embedding table must actually have
        return self

    # ---- save / load (so the tokenizer is trained once and then reused) ----
    def save(self, path, meta=None):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab_size": self.vocab_size, "merges": self.merges, "itos": self.itos,
                       "meta": meta or {}}, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(data["vocab_size"])
        tok.merges = [list(m) for m in data["merges"]]
        tok.itos = list(data["itos"])
        tok.stoi = {t: i for i, t in enumerate(tok.itos)}
        tok._ranks = {(a, b): r for r, (a, b) in enumerate(tok.merges)}
        tok._id_cache = {}
        tok.vocab_size = len(tok.itos)
        return tok

    # ---- inference ----
    def _tokenize_word(self, word):
        syms = self._word_symbols(word)
        while len(syms) > 1:
            best_rank, best_j = None, -1
            for j in range(len(syms) - 1):
                r = self._ranks.get((syms[j], syms[j + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_j = r, j
            if best_rank is None:
                break
            syms = self._merge(syms, (syms[best_j], syms[best_j + 1]))
        return syms

    def _word_ids(self, word):
        ids = self._id_cache.get(word)
        if ids is None:
            unk = self.stoi["<unk>"]
            ids = [self.stoi.get(t, unk) for t in self._tokenize_word(word)]     # symbols never seen in training -> <unk>
            self._id_cache[word] = ids
        return ids

    def tokenize(self, text):
        out = []
        for word in str(text).split():
            out.extend(self._tokenize_word(word))
        return out

    def get_vocab_size(self):
        return len(self.itos)

    def encode(self, text):
        out = []
        for word in str(text).split():
            out.extend(self._word_ids(word))
        return out

    def encode_file(self, path, block_chars=5_000_000):
        """Encode a whole text file into a packed array of ids, block by block (bounded memory)."""
        out = array.array("i")
        leftover = ""
        with open(path, "r", encoding="utf-8") as f:
            while True:
                block = f.read(block_chars)
                if not block:
                    break
                block = leftover + block
                parts = block.split()
                if parts and not block[-1].isspace():
                    leftover = parts.pop()
                else:
                    leftover = ""
                cache = self._id_cache
                for word in parts:
                    ids = cache.get(word)
                    if ids is None:
                        ids = self._word_ids(word)
                    out.extend(ids)
        if leftover:
            out.extend(self._word_ids(leftover))
        return out

    def decode(self, ids):
        """ids -> string. Words are re-joined with single spaces; <unk> decodes to '?'."""
        words, cur = [], bytearray()
        for i in ids:
            i = int(i)
            if not 0 <= i < len(self.itos):
                continue
            tok = self.itos[i]
            if tok in ("<pad>", "<s>", "</s>"):
                continue
            if tok == "<unk>":
                cur.extend(b"?")
                continue
            ends_word = tok.endswith(_END)
            if ends_word:
                tok = tok[: -len(_END)]
            cur.extend(ord(c) & 0xFF for c in tok)
            if ends_word:
                words.append(bytes(cur).decode("utf-8", errors="replace"))
                cur = bytearray()
        if cur:
            words.append(bytes(cur).decode("utf-8", errors="replace"))
        return " ".join(words)


