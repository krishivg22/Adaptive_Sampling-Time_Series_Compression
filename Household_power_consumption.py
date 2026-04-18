import requests
import numpy as np
from collections import Counter
import heapq
from scipy.fftpack import dct, idct
import pywt

URL = "https://api.open-meteo.com/v1/forecast?latitude=28.67&longitude=77.22&hourly=temperature_2m"
js = requests.get(URL).json()
print(type(js))
print(js.keys())
SCALE = 100

def load_data(url):
    js = requests.get(url).json()
    values = js["hourly"]["temperature_2m"]
    return np.array(values, dtype=np.float64)

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def original_bits(x):
    return x.size * 64

def delta_encode(x):
    return x[0], np.diff(x)

def delta_decode(first, d):
    return np.concatenate([[first], first + np.cumsum(d)])

def rle_encode(arr):
    values, counts = [], []
    prev, count = arr[0], 1
    for x in arr[1:]:
        if x == prev:
            count += 1
        else:
            values.append(prev)
            counts.append(count)
            prev, count = x, 1
    values.append(prev)
    counts.append(count)
    return np.array(values), np.array(counts)

def rle_decode(values, counts):
    return np.repeat(values, counts)

class Node:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_codes(data):

    if len(data) == 0:
        return {}

    freq = Counter(data)

    # special case: only one symbol
    if len(freq) == 1:

        symbol = list(freq.keys())[0]
        return {symbol: "0"}

    heap = [Node(f, s) for s, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:

        a = heapq.heappop(heap)
        b = heapq.heappop(heap)

        heapq.heappush(
            heap,
            Node(a.freq + b.freq, left=a, right=b)
        )

    root = heap[0]

    codes = {}

    def dfs(node, code=""):

        if node.symbol is not None:
            codes[node.symbol] = code
            return

        dfs(node.left, code+"0")
        dfs(node.right, code+"1")

    dfs(root)

    return codes

def huffman_encode(data, codes):
    return "".join(codes[x] for x in data)

def huffman_decode(bitstring, codes):
    rev = {v: k for k, v in codes.items()}
    out, cur = [], ""
    for b in bitstring:
        cur += b
        if cur in rev:
            out.append(rev[cur])
            cur = ""
    return np.array(out)

def dct_compress(x, keep_ratio=0.1):
    coeff = dct(x, norm='ortho')
    k = int(len(coeff) * keep_ratio)
    c = np.zeros_like(coeff)
    c[:k] = coeff[:k]
    return idct(c, norm='ortho'), k

def piecewise_poly_compress(x, window=128, degree=2):
    coeffs = []
    for i in range(0, len(x), window):
        seg = x[i:i+window]
        t = np.arange(len(seg))
        c = np.polyfit(t, seg, degree) if len(seg) > degree else np.zeros(degree+1)
        coeffs.append((len(seg), c))
    return coeffs

def piecewise_poly_decompress(coeffs):
    out = []
    for length, c in coeffs:
        out.append(np.polyval(c, np.arange(length)))
    return np.concatenate(out)

def paa_compress(x, window=128):
    paa = np.array([np.mean(x[i:i+window]) for i in range(0, len(x), window)])
    return paa, window

def paa_decompress(paa, window, n):
    return np.repeat(paa, window)[:n]

def wavelet_compress(x, wavelet='db4', keep_ratio=0.1):
    coeffs = pywt.wavedec(x, wavelet)
    flat = np.concatenate([c.ravel() for c in coeffs])
    k = int(len(flat) * keep_ratio)
    idx = np.argsort(np.abs(flat))[-k:]
    mask = np.zeros_like(flat)
    mask[idx] = flat[idx]
    rebuilt, pos = [], 0
    for c in coeffs:
        rebuilt.append(mask[pos:pos+c.size])
        pos += c.size
    recon = pywt.waverec(rebuilt, wavelet)
    return recon[:len(x)], k

def train_ar_model(x, p):
    X, Y = [], []

    for i in range(p, len(x)):
        X.append(x[i-p:i])
        Y.append(x[i])

    X = np.array(X)
    Y = np.array(Y)

    w = np.linalg.lstsq(X, Y, rcond=None)[0]
    return w


def forecast_ar(train, p):
    w = train_ar_model(train, p)

    preds = list(train[:p])   # first p values remain original

    for t in range(p, len(train)):
        x_input = np.array(preds[t-p:t])   # only predicted past values
        preds.append(np.dot(w, x_input))

    return np.array(preds)

def split_segments(x, size):
    return [x[i:i+size] for i in range(0, len(x), size)]

def entropy_int(arr):
    if len(arr) == 0:
        return 0
    
    vals, counts = np.unique(arr, return_counts=True)
    p = counts / len(arr)
    
    return -np.sum(p * np.log2(p))

def adaptive_compress_simple(train, segment_size=240):

    segments = split_segments(train, segment_size)

    variability = [
        np.std(seg)
        for seg in segments
    ]

    threshold = np.percentile(variability, 30)

    compressed = []

    for seg, v in zip(segments, variability):

        # smooth region → more compression
        if v <= threshold:

            rec, k = wavelet_compress(
                seg,
                keep_ratio=0.1
            )

            compressed.append(
                ("WAVELET", (rec, k))
            )

        # complex region → preserve detail
        else:

            first, d = delta_encode(seg)

            d_scaled = np.round(
                d * SCALE
            ).astype(np.int32)

            codes = build_huffman_codes(d_scaled)

            bits = huffman_encode(
                d_scaled,
                codes
            )

            compressed.append(
                ("HUFFMAN", (first, bits, codes))
            )

    return compressed, len(train)
def adaptive_decompress_simple(compressed, total_len):

    reconstructed = []

    for method, data in compressed:

        if method == "WAVELET":

            rec, _ = data

        else:

            first, bits, codes = data

            d = huffman_decode(bits, codes) / SCALE

            rec = delta_decode(first, d)

        reconstructed.append(rec)

    return np.concatenate(reconstructed)[:total_len]

def adaptive_size_simple(compressed):

    bits = 0

    for method, data in compressed:

        if method == "WAVELET":

            _, k = data

            bits += k * 64

        else:

            _, bits_str, _ = data

            bits += len(bits_str) + 64

    return bits

def adaptive_compress_forecast_2method(train, segment_size=240):

    # predict future using AR model
    pred = forecast_ar(train, p=20)

    segments = split_segments(train, segment_size)

    pred_segments = split_segments(pred, segment_size)

    # compute prediction error per segment
    errors = [
        rmse(s, p)
        for s,p in zip(segments, pred_segments)
    ]

    threshold = np.percentile(errors, 30)

    compressed = []

    for seg, err in zip(segments, errors):

        # predictable region
        if err <= threshold:

            rec,k = wavelet_compress(
                seg,
                keep_ratio=0.1
            )

            compressed.append(
                ("WAVELET",(rec,k))
            )

        # unpredictable region
        else:

            first,d = delta_encode(seg)

            d_scaled = np.round(
                d*SCALE
            ).astype(np.int32)

            codes = build_huffman_codes(d_scaled)

            bits = huffman_encode(
                d_scaled,
                codes
            )

            compressed.append(
                ("HUFFMAN",(first,bits,codes))
            )

    return compressed,len(train)
def adaptive_decompress_forecast_2method(compressed,total_len):

    reconstructed = []

    for method,data in compressed:

        if method == "WAVELET":

            rec,_ = data

        else:

            first,bits,codes = data

            d = huffman_decode(bits,codes)/SCALE

            rec = delta_decode(first,d)

        reconstructed.append(rec)

    return np.concatenate(reconstructed)[:total_len]
def main():
    print("Loading data...")
    H = load_data(URL)
    base_bits = original_bits(H)
    results = []

    first, d = delta_encode(H)
    results.append(("Delta", 1.0, 0.0))

    d_scaled = np.round(d * SCALE).astype(np.int32)
    v, c = rle_encode(d_scaled)
    H_rec = delta_decode(first, rle_decode(v, c) / SCALE)
    results.append(("Delta+RLE", (v.size + c.size)*32 / base_bits, rmse(H, H_rec)))

    codes = build_huffman_codes(d_scaled)
    bits = huffman_encode(d_scaled, codes)
    H_rec = delta_decode(first, huffman_decode(bits, codes) / SCALE)
    results.append(("Delta+Huffman", (len(bits)+64)/base_bits, rmse(H, H_rec)))

    H_rec, k = dct_compress(H)
    results.append(("DCT", k*64/base_bits, rmse(H, H_rec)))

    coeffs = piecewise_poly_compress(H)
    H_rec = piecewise_poly_decompress(coeffs)
    poly_bits = sum(len(c)*64 + 32 for _, c in coeffs)
    results.append(("Piecewise Poly", poly_bits/base_bits, rmse(H, H_rec)))

    paa, w = paa_compress(H)
    H_rec = paa_decompress(paa, w, len(H))
    results.append(("PAA", paa.size*64/base_bits, rmse(H, H_rec)))

    H_rec, k = wavelet_compress(H)
    results.append(("Wavelet", k*64/base_bits, rmse(H, H_rec)))

    train = H
    
    compressed, n = adaptive_compress_simple(
        train,
        segment_size=60
    )
    
    H_rec = adaptive_decompress_simple(
        compressed,
        n
    )
    
    bits_used = adaptive_size_simple(compressed)
    
    results.append((
        "Adaptive (2-method)",
        bits_used / base_bits,
        rmse(H, H_rec)
    ))
    compressed,n = adaptive_compress_forecast_2method(
    train,segment_size=60
)

    H_rec = adaptive_decompress_forecast_2method(
        compressed,
        n
    )
    
    bits_used = adaptive_size_simple(compressed)
    
    results.append((
        "Adaptive Forecast (2-method)",
        bits_used/base_bits,
        rmse(H,H_rec)
    ))
    
    print("\nRESULTS : Temperature / Power dataset")
    print("="*65)
    print("Method                    | Compression | RMSE")
    print("="*65)
    for n, r, e in results:
        print(f"{n:26} | {r:11.4f} | {e:.6f}")

if __name__ == "__main__":
    main()