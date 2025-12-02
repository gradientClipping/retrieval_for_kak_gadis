import streamlit as st
import torch
import numpy as np
import re
import math
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
from functools import lru_cache
from itertools import product
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================

# --- Hugging Face Configuration ---
# This points to the adapter you uploaded to Hugging Face
ADAPTER_REPO_ID = "juangwijaya/qwen4b-embed-lora-pas"

# IMPORTANT: Ensure this matches exactly the base model you used for training.
# If "Qwen/Qwen3-Embedding-4B" is a private repo or local name, ensure you have access.
# If you used a public Qwen model (like Qwen2.5-7B or 1.5B), update this ID.
BASE_MODEL_ID = "Qwen/Qwen3-Embedding-4B" 


DOMAIN_TERMS_MULTI = [
    "pinjaman atur sendiri",
    "ovo nabung",
    "push notifikasi",
    "tabungan utama",
    "rekening utama",
    "super care",
    "customer service",
    "call center",
    "tidak bisa",
    "tidak muncul",
    "kopi kenangan",
    "kartu untung",
    "shopee pay",
    "ayam untung"
]

DOMAIN_TERMS_SINGLE = {
    "superbank", "pas", "celengan", "deposito", "qris",
    "pinjaman", "sendiri", "limit", "antrian", "saku", "rewards", "promo",
    "grab", "tabungan", "bank", "whatsapp", "connect", "koneksi",
    "hubung", "hubungkan", "minta", "beranda", "dashboard", "aktivasi",
    "registrasi", "notifikasi", "request", "grabfood", "grabbike", "grabcar", "shell", "watson",
    "ovo", "bunga", "nama", "saqu", "bca", "bri", "mandiri", "monee", "seabank"
}

ID_LEXICON = {
    "yang", "untuk", "dengan", "di", "ke", "dari", "bisa", "tidak", "akan",
    "sudah", "belum", "lagi", "mohon", "tolong", "harap", "kalau", "kalo",
    "supaya", "agar", "karena", "dan", "atau", "jadi", "saja", "aja", "lah",
    "nih", "sih", "dong", "deh", "loh", "ya", "nggak", "ga", "enggak", "ok",
    "oke", "baik", "terima", "kasih", "makasih", "yaudah", "begitu", "kayak",
    "maksud", "artinya", "tentang", "soalnya", "barusan", "kemarin", "tadi",
    "sekarang", "nanti", "besok", "susah", "minta", "meminta", "naikkin", "doang",
    "nelpon", "telepon", "habis", "kali", "kalau", "konfirmasi", "informasi", "sosialisasi",
    "pilihan", "cicilan", "enggak", "ga", "sign", "berbunga", "approve", "approval", "setujui", "padahal",
    
    "halo", "hai", "kak", "kakak", "nasabah", "customer", "pengguna",
    "anda", "aku", "saya", "kita", "kami", "team", "tim", "superbot", "cs",
    "agent", "superbank", "di", "super", "saham",
    
    "apa", "siapa", "kapan", "dimana", "bagaimana", "gimana", "kenapa",
    "mengapa", "apakah", "mana", "kok", "loh", "ngapa", "berapa",
    
    "ktp", "ektp", "nik", "npwp", "paspor", "foto", "selfie",
    "nomor", "no", "rekening", "akun", "login", "akses", "verifikasi",
    "otp", "kode", "token", "password", "sandi", "email",
    
    "saldo", "uang", "tarik", "menarik", "transfer", "kirim", "isi", "topup",
    "bayar", "pembayaran", "tagihan", "angsuran", "tenor", "bunga",
    "biaya", "gratis", "cashback", "promo", "limit", "naik", "turun",
    "pencairan", "penarikan", "pengajuan", "disetujui", "tolak", "ditolak",
    "persetujuan", "pembatalan", "gagal", "berhasil", "proses", "cair", "mengajukan", "mau", "sebelumnya",

    "pinjaman", "ovo", "nabung", "celengan", 
    "saku", "deposito", "qris", "debit", "kartu", "virtual", 
    "rekening", "tabungan", "utama", "tambahan", "daftar",

    "aktif", "nonaktif", "diaktifkan", "dinonaktifkan", "status",
    "blokir", "pemblokiran", "tutup", "penutupan", "buka", "dibuka",
    "daftar", "pendaftaran", "login", "logout", "keluar", "masuk",
    "reset", "ubah", "ganti", "hapus", "menabung", "nabung", "akun",

    "error", "bug", "hang", "lemot", "lambat", "tidak", "bisa", "susah",
    "muncul", "keluar", "jalan", "berfungsi", "kendala", "masalah", 
    "gangguan", "trouble", "issue", "lapor", "pengaduan", "mengadu", "sedang", "lagi", "beras",
    "tanyakan", "hubungi", "help", "bantuan", "support", "respon", "respons", "hilang", "nya", "bos", "kakak", "saya", "mengapa", "apa"
    "tanya", "bertanya", "terkait", "baru", "misal", "tanya", "harus", "ajuin",

    "notifikasi", "push", "pesan", "whatsapp", "email", "sms", "connect",
    "pemberitahuan", "informasi", "berita", "pengumuman", "koneksi",
    "hubung", "hubungkan", "cara", "keluhan", "ada",

    "dokumen", "syarat", "ketentuan", "perjanjian", "akta", "formulir",
    "upload", "unggah", "lampiran", "foto", "scan",

    "layanan", "fitur", "menu", "pilihan", "opsi", "versi", "update",
    "aplikasi", "app", "mobile", "tampilan", "interface", "beranda",
    "dashboard", "halaman",

    "waktu", "jam", "hari", "tanggal", "bulan", "tahun", "sementara",
    "cepat", "lama", "sebentar", "menunggu", "tunggu", "antrian", "queue",

    "ojk", "izin", "legal", "keamanan", "privasi", "data", "informasi",
    "sumber", "persetujuan", "disetujui", "penolakan", "perlindungan",

    "versi", "update", "upgrade", "aplikasi", "app", "device", "hp", "ponsel",
    "ios", "android", "browser", "web", "link", "tautan", "url", "kode", "qr", "keringanan",
    "scan", "kamera", "akses", "bebas", "simpan", "aja"

    "rating", "penilaian", "feedback", "ulasan", "komentar", "saran",
    "keluhan", "terima", "kasih", "makasih", "puas", "mudah", "susah", "sudah", "oper", "taruh",
 
    "jakarta", "bandung", "surabaya", "medan", "semarang", "para", "sign", "ini", "isi", "makanan", "aplikasinya", "harganya", "makro", "ekonomi", "mahal", "mahar", "emas",
    "setuju", "mengantre", ""

    "baru", "lama", "pertama", "kedua", "ketiga", "semua", "setiap", "pinjam",
    "beberapa", "banyak", "sedikit", "cukup", "terlalu", "sangat", "bikin", "susah",
    "paling", "lebih", "kurang", "sama", "berbeda", "lain", "dulu", "kredit", "macet", "laporkan", "OJK", "SLIK", "berat",
    "sendiri", "bersama", "sekali", "dua", "tiga", "darurat", "statement", "bank", "tersedia", "account", "mengendap",
    "menonaktifkan", "mengaktifkan", "walaupun", "download", "pecahin", "semoga", "jika", "daftar", "mendaftarkan", "dihubungkan", "menyambungkan", "pemotongan", "potong",
    "keterima", "pakai", "males", "make", "pinjam", "unlink", "malah", "link", "dana", "kan", "awal", "angsuran", "terhubung", "hubungkan", "ribet", "selalu", "lancar", "tanpa", "paylater"
}

COMMON_ALIASES = {
    "pinjamna": "pinjaman",
    "atut": "atur",
    "ovonabung": "ovo nabung",
    "celenan": "celengan",
    "celngan": "celengan",
    "qirish": "qris",
    "kris": "qris",
    "superbang": "superbank",
    "pas": "pinjaman atur sendiri",
    "td": "deposito",
    "tf": "transfer",
    "knp": "kenapa",
    "ga": "tidak",
    "gak": "tidak",
    "gk": "tidak",
    "tdk": "tidak",
    "tdm": "tidak",
    "udh": "sudah",
    "udah": "sudah",
    "blm": "belum",
    "blum": "belum",
    "bs": "bisa",
    "bsa": "bisa",
    "gmn": "bagaimana",
    "bgmn": "bagaimana",
    "kk": "kakak",
    "min": "admin",
    "mins": "admin",
    "cs": "super care",
    "fb": "facebook",
    "wa": "whatsapp",
    "hp": "handphone",
    "tlg": "tolong",
    "tlng": "tolong",
    "mhn": "mohon",
    "mksh": "makasih",
    "tq": "terima kasih",
    "thx": "terima kasih",
    "kpn": "kapan",
    "bdg": "bandung",
    "jkt": "jakarta",
    "sby": "surabaya",
    "ayam nabung": "celengan",
    "ayam cuan": "celengan",
    "nik": "nomor induk kependudukan",
    "mint": "minta",
    "mnta": "minta",
    "tranfer": "transfer",
    "trasfer": "transfer",
    "deposito": "deposito",
    "depo": "deposito",
    "notif": "notifikasi",
    "nggak": "tidak",
    "ngga": "tidak",
    "dmn": "di mana",
    "kmn": "ke mana",
    "knp": "kenapa",
    "sdh": "sudah",
    "msh": "masih",
    "ngga": "tidak",
    "dmn": "di mana",
    "kmn": "ke mana",
    "knp": "kenapa",
    "sdh": "sudah",
    "msh": "masih",
    "blm": "belum",
    "hrs": "harus",
    "bntar": "sebentar",
    "lgsg": "langsung",
    "yg": "yang",
    "sy": "saya",
    "tmn": "teman",
    "stlh": "setelah",
    "lgi": "lagi",
    "dr": "dari",
    "slmt": "selamat",
    "thx": "terima kasih",
    "brgkt": "berangkat",
    "hrus": "harus",
    "ck": "cek",
    "jgn": "jangan",
    "sm": "sama",
    "blm": "belum",
    "gmn": "bagaimana",
    "bgs": "bagus",
    "gbs": "gak bisa",
    "tp": "tapi",
    "wktu": "waktu",
    "ctk": "cetak",
    "bt": "buat",
    "bsa": "bisa",
    "kt": "kita",
    "akunnya": "akun-nya",
    "setelah": "stlh",
    "reff": "referensi",
    "sbank": "superbank",
    "spbank": "superbank",
    "spbang": "superbank",
    "ling": "link",
    "k": "ke",
    "trs": "terus",
    "y": "ya",
    "sya": "saya",
    "sy": "saya",
    "mf": "maaf",
    "dk": "tidak",
    "g": "tidak",
    "klw": "kalau",
    "kl": "kalau",
    "apk": "aplikasi",
    "jdi": "jadi",
    "mlh": "malah",
    "bnr": "benar",
    "kmn2": "kemana-mana",
    "ap2": "apa-apa",
    "gmn2": "gimana-gimana",
    "aq": "saya",
    "pdhl": "padahal",
    "dpt": "dapat",
    "dgn": "dengan"
}

COMMON_TYPO_PATTERNS = {
    "cnnect": "connect",
    "conect": "connect",
    "connct": "connect",
    "tranfer": "transfer",
    "trnasfer": "transfer",
    "pinjman": "pinjaman",
    "pinjamn": "pinjaman",
    "tabunga": "tabungan",
    "tabungn": "tabungan",
    "dbantu": "dibantu",
    
    "depoaito": "deposito",
    "celengsn": "celengan",
    "celengam": "celengan",
    "qeris": "qris",
    "qros": "qris",
    
    "suddah": "sudah",
    "beluum": "belum",
    "bisaa": "bisa",
    "tidakk": "tidak",
    
    "pnijaman": "pinjaman",
    "trnasfer": "transfer",
    "celeagn": "celengan",
    
    "susha": "susah",
    "susa": "susah",
    "cairin": "dicairkan",
    "naikkin": "naikkan",
    "datar": "daftar",
    "stiap": "setiap",
    "lagy": "lagi",
    "sib": "sih"
}

WORD_FREQ = {
    "superbank": 1000,
    "pinjaman": 900,
    "pinjaman atur sendiri": 900,
    "pas":800,
    "transfer": 850,
    "saldo": 800,
    "rekening": 750,
    "tabungan": 700,
    "bunga": 800,
    "bank": 800,
    "harus": 600,
    "bisa": 650,
    "minta": 600,
    "tidak": 600,
    "sudah": 550,
    "belum": 500,
    "bayar": 480,
    "tarik": 460,
    "kirim": 440,
    "pinjam": 800,

    "pas": 500,
    "celengan": 450,
    "qris": 450,
    "deposito": 450,
    "saku": 500,
    "ovo nabung": 400,
    "grab": 400,

    "yang": 400,
    "untuk": 380,
    "dengan": 350,
    "dari": 330,
    "kenapa": 320,
    "bagaimana": 300,
    "mohon": 280,
    "tolong": 280,
    "beranda": 250,
    "connect": 240,
    "hubung": 220,

    "customer": 200,
    "service": 200,
    "super care": 180,
    "bantuan": 180,
    "help": 160,
    
    "masalah": 150,
    "gagal": 140,
    "error": 130,
    "susah": 120,
}

COMMON_BIGRAMS = {
    ("minta", "tolong"): 100,
    ("tidak", "bisa"): 150,
    ("tidak", "muncul"): 120,
    ("sudah", "transfer"): 110,
    ("belum", "masuk"): 100,
    ("cara", "atur"): 100,
    ("kenapa", "tidak"): 85,
    ("gimana", "cara"): 80,
    ("mohon", "bantu"): 75,
    ("tolong", "bantu"): 75,
    ("mau", "tanya"): 70,
    ("customer", "service"): 140,
    ("call", "center"): 130,
    ("ovo", "nabung"): 120,
    ("bunga", "pinjaman"): 120,
    ("bagaimana", "cara"): 120,
    ("fitur", "pas"): 120,
    ("fitur", "pinjaman"): 120,
    ("cairin", "celengan"): 120,
    ("buka", "rekening"): 120,
    ("proses", "pendaftaran"): 120,
    ("sign", "in"): 120,
    ("ajuin", "paylater"): 120,
    ("rekening", "utama"): 120,
    ("tabungan", "utama"): 120,
    ("tidak", "bisa"): 120,
    ("susah", "cairin"): 120,
    ("memunculkan", "pinjaman"): 120,
    ("tidak", "dapat"): 120,
    ("cara", "pinjam"): 120,
    ("cara", "apply"): 120,
    ("cara", "pakai"): 120,
    ("super", "care"): 120,
    ("bunga", "deposito"): 120,
    ("bunga", "celengan"): 120,
    ("bunga", "saku"): 120,
    ("hari", "ini"): 120,
    ("fitur", "deposito"): 120,
    ("fitur", "saku"): 120,
    ("kartu", "untung"): 120,
    ("fitur", "deposito"): 120,
    ("mau", "pinjaman"): 120,
    ("cara", "uninstall"): 120,
    ("cara", "nabung"): 120,
    ("fitur", "deposito"): 120,
    ("minta", "keringanan"): 120,
    ("ayam", "untung"): 120
}

WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9_@.\-:/]+", flags=re.UNICODE)
NUMBER_RE = re.compile(r"\d+([.,]\d+)*")
URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
ALPHANUMERIC_RE = re.compile(r"[A-Za-z]{2,}\d{2,}")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")

# ==========================================
# 2. SPELLING CORRECTION CLASSES
# ==========================================

def is_protected_token(tok: str, check_aliases: bool = True) -> bool:
    if NUMBER_RE.fullmatch(tok): return True
    if URL_RE.match(tok): return True
    if EMAIL_RE.match(tok): return True
    if ALPHANUMERIC_RE.fullmatch(tok): return True
    if check_aliases and tok.lower() in COMMON_ALIASES: return False
    if len(tok) <= 2: return True
    return False

def normalize_token(text: str) -> str:
    text = text.lower()
    text = REPEATED_CHAR_RE.sub(r"\1\1", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@lru_cache(maxsize=10000)
def damerau_levenshtein(s1: str, s2: str) -> int:
    len1, len2 = len(s1), len(s2)
    if abs(len1 - len2) > 3: return abs(len1 - len2)
    d = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1): d[i][0] = i
    for j in range(len2 + 1): d[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)
    return d[len1][len2]

def normalized_edit_distance(a: str, b: str) -> float:
    if not a and not b: return 0.0
    dist = damerau_levenshtein(a, b)
    return dist / max(len(a), len(b), 1)

def char_ngram_similarity(s1: str, s2: str, n: int = 2) -> float:
    def get_ngrams(s, n): return set(s[i:i+n] for i in range(len(s)-n+1))
    if len(s1) < n or len(s2) < n: return 0.0
    ng1, ng2 = get_ngrams(s1, n), get_ngrams(s2, n)
    if not ng1 or not ng2: return 0.0
    intersection = len(ng1 & ng2)
    union = len(ng1 | ng2)
    return intersection / union if union > 0 else 0.0

def indonesian_phonetic_key(word: str) -> str:
    word = word.lower()
    word = word.replace('kh', 'k').replace('sy', 's').replace('ch', 's').replace('ng', 'n')
    if len(word) > 1:
        first_char = word[0]
        rest = re.sub(r'[aeiou]', '', word[1:])
        word = first_char + rest
    return word[:5]

def get_adaptive_max_distance(word_length: int) -> int:
    if word_length <= 3: return 1
    elif word_length <= 6: return 2
    else: return 3

def lock_phrases(text: str, phrases: List[str]) -> str:
    result = text
    for phrase in sorted(phrases, key=len, reverse=True):
        pattern = r'\b' + re.escape(phrase) + r'\b'
        replacement = phrase.replace(" ", "_")
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result

def unlock_phrases(text: str) -> str:
    return text.replace("_", " ")

@dataclass
class CorrectionMeta:
    changed: bool
    confidence: float
    rule: str
    original: str
    candidates: Optional[List[Tuple[str, float]]] = None
    edit_distance: int = 0

class VocabularyIndex:
    def __init__(self):
        self.vocab: Set[str] = set()
        self.by_length: Dict[int, Set[str]] = {}
        self.phonetic_index: Dict[str, Set[str]] = {}
        self._build_index()
    
    def _build_index(self):
        self.vocab |= ID_LEXICON
        self.vocab |= DOMAIN_TERMS_SINGLE
        for phrase in DOMAIN_TERMS_MULTI:
            for token in phrase.split():
                cleaned = re.sub(r"[^\w]", "", token.lower())
                if cleaned: self.vocab.add(cleaned)
        self.vocab |= set(COMMON_ALIASES.values())
        for word in self.vocab:
            length = len(word)
            if length not in self.by_length: self.by_length[length] = set()
            self.by_length[length].add(word)
            phon_key = indonesian_phonetic_key(word)
            if phon_key not in self.phonetic_index: self.phonetic_index[phon_key] = set()
            self.phonetic_index[phon_key].add(word)
    
    def get_candidates(self, token: str, max_distance: int = None) -> List[str]:
        if max_distance is None: max_distance = get_adaptive_max_distance(len(token))
        token_len = len(token)
        candidates = set()
        for length in range(max(1, token_len - max_distance), token_len + max_distance + 1):
            if length in self.by_length:
                for word in self.by_length[length]:
                    dist = damerau_levenshtein(token, word)
                    if dist <= max_distance: candidates.add(word)
        phon_key = indonesian_phonetic_key(token)
        if phon_key in self.phonetic_index:
            for word in self.phonetic_index[phon_key]:
                dist = damerau_levenshtein(token, word)
                if dist <= max_distance + 1: candidates.add(word)
        return list(candidates)

class SpellingCorrector:
    def __init__(self, confidence_threshold=0.4, max_edit_distance=2, enable_aliases=True, preserve_case=True, use_context=True):
        self.vocab_index = VocabularyIndex()
        self.conf_threshold = confidence_threshold
        self.max_edit_dist = max_edit_distance
        self.enable_aliases = enable_aliases
        self.preserve_case = preserve_case
        self.use_context = use_context
    
    def _score_candidate(self, source, candidate, prev_token=None, next_token=None):
        ned = normalized_edit_distance(source, candidate)
        freq = WORD_FREQ.get(candidate, 1)
        freq_penalty = 1.0 / math.log(freq + 2)
        domain_bonus = 0.15 if candidate in DOMAIN_TERMS_SINGLE else 0.0
        ngram_sim = char_ngram_similarity(source, candidate, n=2)
        ngram_bonus = ngram_sim * 0.2
        phonetic_bonus = 0.1 if indonesian_phonetic_key(source) == indonesian_phonetic_key(candidate) else 0.0
        context_bonus = 0.0
        if self.use_context:
            if prev_token and (prev_token, candidate) in COMMON_BIGRAMS: context_bonus += 0.25
            if next_token and (candidate, next_token) in COMMON_BIGRAMS: context_bonus += 0.25
        return ned + (freq_penalty * 0.08) - domain_bonus - ngram_bonus - phonetic_bonus - context_bonus

    def _calculate_confidence(self, ranked_candidates, source_token):
        if not ranked_candidates: return 0.0
        best_candidate, best_score = ranked_candidates[0]
        if len(ranked_candidates) == 1:
            edit_dist = damerau_levenshtein(source_token, best_candidate)
            return 0.9 if edit_dist == 1 else (0.75 if edit_dist == 2 else 0.6)
        second_score = ranked_candidates[1][1]
        separation = second_score - best_score
        edit_dist = damerau_levenshtein(source_token, best_candidate)
        edit_confidence = 1.0 - (edit_dist / max(len(source_token), 1))
        domain_multiplier = 1.3 if best_candidate in DOMAIN_TERMS_SINGLE else 1.0
        return min(1.0, ((separation * 2.5) + (edit_confidence * 0.3)) * domain_multiplier)

    def correct_token(self, token, prev_token=None, next_token=None):
        original = token
        if self.enable_aliases:
            norm_alias = normalize_token(token)
            alias_key = re.sub(r"[^\w]", "", norm_alias.lower())
            if alias_key in COMMON_ALIASES:
                correction = COMMON_ALIASES[alias_key]
                if self.preserve_case and original[:1].isupper(): correction = correction.capitalize()
                return correction, CorrectionMeta(changed=True, confidence=0.95, rule="alias", original=original)

        if is_protected_token(token): return original, CorrectionMeta(changed=False, confidence=1.0, rule="protected", original=original)
        normalized = normalize_token(token)
        if normalized in self.vocab_index.vocab: return original, CorrectionMeta(changed=False, confidence=1.0, rule="in_vocab", original=original)
        if normalized in COMMON_TYPO_PATTERNS:
            correction = COMMON_TYPO_PATTERNS[normalized]
            if self.preserve_case and original[:1].isupper(): correction = correction.capitalize()
            return correction, CorrectionMeta(changed=True, confidence=0.98, rule="typo_pattern", original=original)
            
        candidates = self.vocab_index.get_candidates(normalized, self.max_edit_dist)
        if not candidates: return original, CorrectionMeta(changed=False, confidence=0.0, rule="no_candidates", original=original)
        
        scored = [(c, self._score_candidate(normalized, c, prev_token, next_token)) for c in candidates]
        ranked = sorted(scored, key=lambda x: x[1])
        confidence = self._calculate_confidence(ranked, normalized)
        best_candidate, _ = ranked[0]
        edit_dist = damerau_levenshtein(normalized, best_candidate)
        
        if confidence >= self.conf_threshold and edit_dist <= self.max_edit_dist:
            correction = best_candidate
            if self.preserve_case and original[:1].isupper(): correction = correction.capitalize()
            return correction, CorrectionMeta(changed=True, confidence=confidence, rule="ranked", original=original, candidates=ranked[:3], edit_distance=edit_dist)
        
        return original, CorrectionMeta(changed=False, confidence=confidence, rule="low_conf", original=original, candidates=ranked[:3])

    def correct(self, text: str) -> Tuple[str, List[Dict]]:
        text = normalize_token(text)
        locked = lock_phrases(text, DOMAIN_TERMS_MULTI)
        pieces, logs, tokens_list, position = [], [], [], 0
        while position < len(locked):
            match = WORD_RE.match(locked, position)
            if not match:
                pieces.append(locked[position])
                position += 1
                continue
            tokens_list.append((match.group(0), match.start(), match.end()))
            position = match.end()
        result_pieces, last_pos = [], 0
        for idx, (token, start, end) in enumerate(tokens_list):
            if start > last_pos: result_pieces.append(locked[last_pos:start])
            prev_tok = normalize_token(tokens_list[idx-1][0]) if idx > 0 else None
            next_tok = normalize_token(tokens_list[idx+1][0]) if idx < len(tokens_list)-1 else None
            corrected, meta = self.correct_token(token, prev_tok, next_tok)
            result_pieces.append(corrected)
            logs.append({"original": token, "corrected": corrected, "changed": meta.changed, "confidence": meta.confidence, "rule": meta.rule})
            last_pos = end
        if last_pos < len(locked): result_pieces.append(locked[last_pos:])
        return unlock_phrases("".join(result_pieces)), logs

class TextProcessor:
    def __init__(self, correct_spelling=True):
        self.correct_spelling = correct_spelling
        self.spelling_corrector = SpellingCorrector() if correct_spelling else None
    
    def process(self, text: str) -> Dict:
        result = {'original_text': text, 'processed_text': text, 'spelling_corrected': False, 'spelling_logs': []}
        if self.correct_spelling:
            corrected_text, logs = self.spelling_corrector.correct(text)
            result['processed_text'] = corrected_text
            result['spelling_corrected'] = any(log['changed'] for log in logs)
            result['spelling_logs'] = logs
        return result

# ==========================================
# 3. EMBEDDING & SEARCH ENGINE
# ==========================================

@st.cache_resource
def load_hf_model():
    """Loads the tokenizer and model with LoRA adapter from Hugging Face."""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        print(f"Loading Base Model: {BASE_MODEL_ID} on {device}")
        
        # 1. Load Tokenizer (Trust remote code is crucial for Qwen)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        
        # 2. Load Base Model
        base_model = AutoModel.from_pretrained(BASE_MODEL_ID, trust_remote_code=True).to(device)
        
        # 3. Load Adapter from Hugging Face Repo
        print(f"Loading Adapter from HF Repo: {ADAPTER_REPO_ID}")
        try:
            model = PeftModel.from_pretrained(base_model, ADAPTER_REPO_ID)
            model.eval()
            return tokenizer, model, device
        except Exception as e:
            st.error(f"Could not load adapter from Hugging Face ({ADAPTER_REPO_ID}). Error: {str(e)}")
            st.warning("Running without Adapter (Base model only).")
            return tokenizer, base_model, device

    except Exception as e:
        st.error(f"Error loading base model ({BASE_MODEL_ID}): {str(e)}")
        st.info("Check if BASE_MODEL_ID matches the one you used for training.")
        return None, None, None

def get_embedding_local(text: str, tokenizer, model, device) -> np.ndarray:
    """Generates embedding using the loaded local model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()[0]

def sigmoid(x: float) -> float: return 1 / (1 + np.exp(-x))

def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if len(scores) == 0: return scores
    scores_shifted = scores / temperature - np.max(scores / temperature)
    exp_scores = np.exp(scores_shifted)
    return exp_scores / np.sum(exp_scores)

def tokenize_bm25(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())

class SearchEngine:
    def __init__(self, alpha=0.4, beta=0.4, gamma=0.2, use_bm25=True, bm25_weight=0.3, temperature=1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_bm25 = use_bm25
        self.bm25_weight = bm25_weight
        self.temperature = temperature
        self.documents = []
        self.embeddings = []
        self.bm25 = None
        
        # Load model once from HF
        self.tokenizer, self.model, self.device = load_hf_model()

    def embed(self, text: str) -> np.ndarray:
        if self.model:
            return get_embedding_local(text, self.tokenizer, self.model, self.device)
        else:
            return np.random.rand(1024).astype(np.float32)

    def index_documents(self, documents: List[str]):
        self.documents = documents
        progress_bar = st.progress(0, text="Indexing Knowledge Base...")
        
        self.embeddings = []
        for i, doc in enumerate(documents):
            self.embeddings.append(self.embed(doc))
            if i % 5 == 0:
                progress_bar.progress((i + 1) / len(documents))
        
        if self.use_bm25:
            tokenized_docs = [tokenize_bm25(doc) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        
        progress_bar.empty()

    def search(self, query_init, query_corrected, top_k=5, retrieval_k=50):
        q_corrected_emb = self.embed(query_corrected)
        
        # Hybrid Retrieval
        dense_scores = np.array([np.dot(q_corrected_emb, d) / (np.linalg.norm(q_corrected_emb) * np.linalg.norm(d)) for d in self.embeddings])
        
        if self.use_bm25:
            sparse_scores = self.bm25.get_scores(tokenize_bm25(query_corrected))
            dense_norm = softmax(dense_scores, temperature=self.temperature)
            sparse_norm = softmax(sparse_scores, temperature=self.temperature)
            combined_scores = (1 - self.bm25_weight) * dense_norm + self.bm25_weight * sparse_norm
        else:
            combined_scores = dense_scores

        candidate_indices = np.argsort(combined_scores)[::-1][:retrieval_k]

        # Reranking
        q_init_emb = self.embed(query_init)
        results = []
        
        for idx in candidate_indices:
            doc_emb = self.embeddings[idx]
            s_init = np.dot(doc_emb, q_init_emb) / (np.linalg.norm(doc_emb) * np.linalg.norm(q_init_emb))
            s_corrected = np.dot(doc_emb, q_corrected_emb) / (np.linalg.norm(doc_emb) * np.linalg.norm(q_corrected_emb))
            d_corrected = np.linalg.norm(doc_emb - q_corrected_emb)
            d_norm = sigmoid(d_corrected)
            
            final_score = (self.alpha * s_init) + (self.beta * s_corrected) - (self.gamma * d_norm)
            
            results.append({
                "document": self.documents[idx],
                "score": float(final_score),
                "details": {
                    "s_init": float(s_init),
                    "s_corrected": float(s_corrected),
                    "d_norm": float(d_norm)
                }
            })
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

# ==========================================
# 4. KNOWLEDGE BASE
# ==========================================
CLEAN_KB = [
    "Nasabah tanya sudah upgrade OVO Nabung tapi masih tidak bisa akses Pinjaman Atur Sendiri",
    "Nasabah tanyakan apakah dengan upgrade ke OVO Nabung, Nasabah dapat masuk ke daftar tunggu Pinjaman Atur Sendiri",
    "Nasabah tanyakan apakah dengan upgrade ke OVO Nabung, Nasabah bisa langsung dapat Pinjaman Atur Sendiri",
    "Nasabah tanyakan berapa lama daftar tunggu untuk bisa daftar Pinjaman Atur Sendiri",
    "Nasabah tidak merasa meminjam tetapi tertera ada pinjaman di SLIK",
    "Nasabah memiliki pinjaman di Easycash/Amartha/Indodana/AdaKami/Rupiah Cepat/Julo/Adapundi/Asetku/UangMe, namun tidak merasa meminjam",
    "Seputar kendala pinjaman di Easycash/Amartha/Indodana/AdaKami/Rupiah Cepat/Julo/Adapundi/Asetku/UangMe",
    "Nasabah komplain terkait ketidaksesuaian pencatatan SLIK untuk pinjaman di loan channeling",
    "Nasabah komplain terkait ketidaksesuaian pencatatan SLIK untuk Pinjaman Atur Sendiri Superbank",
    "Nasabah tanyakan terkait Pinjam Atur Sendiri",
    "Nasabah tanyakan keunggulan rekening Pinjaman Atur Sendiri dari sisi bunga, biaya admin, dll seputar informasi produk Pinjaman Atur Sendiri",
    "Nasabah tanyakan syarat pengajuan Pinjaman Atur Sendiri",
    "Nasabah tanyakan bagaimana cara menemukan menu Pinjaman Atur Sendiri pada aplikasi Superbank",
    "Nasabah tanyakan apakah Pinjaman Atur Sendiri [Pinjaman Atur Sendiri] Superbank aman untuk dana pinjamannya",
    "Nasabah tanyakan cara melakukan pembayaran Pinjaman Atur Sendiri",
    "Nasabah tanyakan sumber rekening yang dapat digunakan untuk pembayaran Pinjaman Atur Sendiri",
    "Nasabah tanyakan apakah auto debit dapat diaktifkan dan nonaktifkan",
    "Nasabah tanyakan apakah saya bisa memilih tanggal auto debit",
    "Nasabah tanyakan berapa banyak jumlah auto debit yang akan ditarik",
    "Nasabah tanyakan sumber rekening auto-debit yang bisa digunakan",
    "Nasabah tanyakan persyaratan Tarik Dana \"Pinjaman Atur Sendiri\" [Pinjaman Atur Sendiri]",
    "Nasabah tanyakan cara mengajukan Pinjaman Atur Sendiri",
    "Nasabah tanyakan cara mencairkan Pinjaman Atur Sendiri",
    "Nasabah tanyakan kenapa perlu melakukan verifikasi wajah saat melakukan pencairan pinjaman",
    "Nasabah tanyakan cara untuk melakukan pembayaran cicilan Pinjaman Atur Sendiri",
    "Nasabah tanya kenapa pengajuan pinjaman ditolak",
    "Nasabah tanyakan status pengajuan Pinjaman Atur Sendiri",
    "Nasabah tanyakan limit, total tagihan, tenor, dll (untuk nasabah yang pinjamannya sudah disetujui)",
    "Nasabah tanyakan cara untuk naik limit atau mengganti tenor cicilan",
    "Nasabah tanyakan seputar biaya bunga",
    "Nasabah tanyakan biaya keterlambatan",
    "Nasabah tanyakan berapa tenornya",
    "Nasabah tanyakan limit pinjaman",
    "Nasabah keluhkan biaya, tenor, limit tidak sesuai ekspektasi",
    "Nasabah tanyakan kenapa hanya bisa pilih tenor cicilan min 3 bulan",
    "Nasabah ingin tutup Pinjaman Atur Sendiri",
    "Nasabah request untuk tutup fasilitas Pinjaman Atur Sendiri",
    "Nasabah meminta pemblokiran limit Pinjaman Atur Sendiri",
    "Nasabah meminta untuk lakukan pencabutan limit Pinjaman Atur Sendiri",
    "Nasabah minta pembukaan blokir limit Pinjaman Atur Sendiri",
    "Nasabah minta buka blokir (status reason pemblokiran 'Under Fraud Investigation')",
    "Nasabah keluhkan pencairan pinjaman gagal",
    "Nasabah tidak bisa lakukan pencairan pinjaman (muncul respon error)",
    "Muncul respon \"Coba Tarik Dana Lagi Nanti yaa\" saat Nasabah lakukan pencairan",
    "Muncul respon \"Mohon bersabar, akunmu dalam proses verifikasi\"",
    "Gagal lakukan pencairan saat proses verifikasi wajah",
    "Nasabah keluhkan mengalami kendala pengajuan pinjaman",
    "Nasabah keluhkan saat mengajukan pinjaman muncul blank screen",
    "Muncul respon 'Maaf lagi ada gangguan'",
    "Nasabah keluhkan tidak menerima dana tapi ada tagihan",
    "Nasabah Ingin pelunasan lebih awal, tanya keringanan pelunasan lebih awal",
    "Nasabah infokan sudah bayar tapi masih ada tagihan",
    "Nasabah keluhkan tidak bisa lakukan pembayaran manual Pinjaman Atur Sendiri",
    "Muncul respon \"maaf lagi ada gangguan\" saat lakukan pembayaran manual Pinjaman Atur Sendiri",
    "Muncul respon \"pembayaran tagihan masih diproses\"",
    "Nasabah keluhkan auto debit pembayaran Pinjaman Atur Sendiri tidak berjalan",
    "Nasabah keluhkan tidak merasa meminjam tapi ada tagihan",
    "Nasabah keluhkan penagihan kasar",
    "EC/kantor tidak ingin dihubungi penagih dan minta dihapuskan",
    "Nasabah sudah ubah nomor telepon, tapi pihak penagihan/collection masih menghubungi ke nomor lama",
    "Tagihan sudah lunas, tapi masih dihubungi pihak penagihan/collection",
    "Keluhan terkait pihak penagihan lainnya",
    "Limit/tenor tidak sesuai ekspektasi, minta naik limit",
    "Nasabah tanya bunga pinjaman",
    "Nasabah infokan pengajuan Pinjaman Atur Sendiri ditolak, kapan bisa coba apply lagi?",
    "Cara mengubah tanggal jatuh tempo",
    "Kenapa tidak bisa memilih/merubah tanggal jatuh tempo untuk pinjaman baru?",
    "Cara mengubah tanggal auto debit",
    "Nasabah tanyakan apakah ada maksimal denda",
    "Nasabah minta keringanan denda/bunga",
    "Nasabah kirimkan surat keringanan denda/bunga",
    "Nasabah kirimkan surat pernyataan/pemberitahuan tidak mampu membayar tagihan Pinjaman Atur Sendiri",
    "Nasabah tanyakan berapa maksimal tenor Pinjaman Atur Sendiri",
    "Nasabah tanya simulasi pinjaman",
    "Nasabah tanya nominal pembayaran setelah pinjaman terlambat",
    "Nasabah tanyakan Apakah Pinjaman Atur Sendiri Nasabah dilaporkan ke SLIK (Sistem Layanan Informasi Keuangan)",
    "Nasabah tanyakan seputar SLIK",
    "Nasabah tanyakan terkait kenapa sudah bayar namun SLIK belum terupdate",
    "Nasabah tanyakan tanggal pelaporan SLIK Superbank",
    "Nasabah tanyakan bagaimana cara menghitung Bunga \"Pinjaman Atur Sendiri\" [Pinjaman Atur Sendiri]",
    "Nasabah minta surat keterangan lunas atas pinjaman Pinjaman Atur Sendiri",
    "Nasabah/Non-Nasabah minta surat keterangan lunas atas pinjaman dari: Mitra penyaluran kredit (Easycash/Amartha/Adakami/Indodana/Rupiah Cepat/Julo/Adapundi/Asetku/Uangme)",
    "Nasabah lakukan pencairan pinjaman, limit sudah berkurang namun dana tidak cair ke tabungan utama (tidak muncul pada riwayat tagihan)",
    "Driver Grab (DAX) tanyakan terkait:\n- Cara bergabung program Rent to Win / Auto Loan / Sahabat Sejati + / Kredit Kendaraan Bermotor Superbank untuk DAX\n- Pinjaman di Superbank hingga 142.5 juta untuk DAX",
    "Nasabah tanyakan pencairan limit Pinjaman Atur Sendiri setelah dilakukan pembayaran",
    "Nasabah tanyakan limit Pinjaman Atur Sendiri yang didapat setelah program selesai",
    "Nasabah yang tergabung ke dalam program tidak mendapatkan pembayaran Pinjaman Atur Sendiri dari Grab",
    "Ketidaksesuaian nominal pembayaran Pinjaman Atur Sendiri oleh Grab",
    "Nasabah yang tergabung ke dalam program ingin melakukan pelunasan dipercepat",
    "Nasabah yang tergabung ke dalam program meminta keringanan pembayaran / restrukturisasi pembayaran",
    "Fitur Pinjaman Atur Sendiri Nasabah telah ditutup/tidak diperpanjang",
    "Nasabah keluhkan tidak bisa lakukan pencairan limit Pinjaman Atur Sendiri (hanya ada informasi tagihan yang harus dibayar)",
    "Respon: \"Pinjaman Atur Sendiri Udah Nggak Tersedia\"",
    "Status di CRM: \"FACILITY_NOT_RENEW\"",
    "Nasabah tanyakan kenapa ada perubahan pada limit Pinjaman Atur Sendirinya",
    "Nasabah tanyakan apakah benar Pinjaman Atur Sendiri akan dikenakan biaya admin/biaya pelunasan dipercepat",
    "Nasabah tanyakan, ajukan Pinjaman Atur Sendiri tapi kenapa diarahkan ke aplikasi lain",
    "Bagaimana cara ajukan pinjaman di aplikasi RupiahCepat/Easycash?",
    "Penawarannya ketutup, bagaimana cara melihat penawarannya lagi?",
    "Bagaimana cara lakukan pembayaran/pencairan pinjaman di RupiahCepat/Easycash?",
    "Kenapa pengajuan pinjaman di RupiahCepat/Easycash ditolak?",
    "Pertanyaan lainnya terkait mitra"
]

# ==========================================
# 5. STREAMLIT APP UI
# ==========================================

st.set_page_config(page_title="KM Assistant - PAS Retrieval", layout="wide")

st.title("Buat User")
st.markdown("""
Buat Ka Gadis.
""")

# --- Sidebar: Controls ---
with st.sidebar:
    st.header("Reranking Weights")
    alpha = st.slider("Alpha (Original Query Similarity)", 0.0, 1.0, 0.425, 0.05)
    beta = st.slider("Beta (Corrected Query Similarity)", 0.0, 1.0, 0.425, 0.05)
    gamma = st.slider("Gamma (Euclidean Distance Penalty)", 0.0, 1.0, 0.15, 0.05)
    
    st.markdown("---")
    st.write(r"**Formula:**")
    st.latex(r"Score = \alpha \cdot S_{init} + \beta \cdot S_{corr} - \gamma \cdot D_{norm}")

# --- Initialize System ---
if 'search_engine' not in st.session_state:
    with st.spinner("Initializing Search Engine & Downloading Model..."):
        # This will now fetch from Hugging Face
        engine = SearchEngine(alpha=alpha, beta=beta, gamma=gamma)
        engine.index_documents(CLEAN_KB)
        st.session_state.search_engine = engine
        st.session_state.processor = TextProcessor(correct_spelling=True)
    st.success("System Initialized! Model loaded from Hugging Face.")

# Update weights dynamically
st.session_state.search_engine.alpha = alpha
st.session_state.search_engine.beta = beta
st.session_state.search_engine.gamma = gamma

# --- Main Interface ---
query = st.text_input("Enter your query:", placeholder="e.g., cara byar pjm atur sndiri")

if st.button("Search") and query:
    # 1. Processing (Spelling)
    processed = st.session_state.processor.process(query)
    corrected_query = processed['processed_text']
    
    st.markdown("### Query Processing")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Original:** {query}")
    with col2:
        if processed['spelling_corrected']:
            st.success(f"**Corrected:** {corrected_query}")
        else:
            st.warning("**No Correction Needed**")

    # Display Spelling Logs if changes occurred
    if processed['spelling_corrected']:
        with st.expander("View Spelling Correction Details"):
            st.table(processed['spelling_logs'])

    # 2. Retrieval
    st.markdown("### Top 5 Results")
    results = st.session_state.search_engine.search(query, corrected_query, top_k=5)
    
    for i, res in enumerate(results, 1):
        with st.container():
            st.markdown(f"**{i}. {res['document']}**")
            
            # Visual Score Bar
            score_pct = max(0, min(1, res['score'])) # Clamp for display
            st.progress(score_pct)
            
            # Details
            cols = st.columns(4)
            cols[0].metric("Final Score", f"{res['score']:.4f}")
            cols[1].metric("Sim (Init)", f"{res['details']['s_init']:.3f}")
            cols[2].metric("Sim (Corr)", f"{res['details']['s_corrected']:.3f}")
            cols[3].metric("Dist Penalty", f"{res['details']['d_norm']:.3f}")
            st.divider()