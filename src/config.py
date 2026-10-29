"""Configuration centrale du projet : chemins, colonnes NSL-KDD, taxonomie des attaques."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"

RANDOM_STATE = 42

# 41 attributs + étiquette + niveau de difficulté (format NSL-KDD)
COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

CAT_COLS = ["protocol_type", "service", "flag"]
FEATURE_COLS = [c for c in COLS if c not in ("label", "difficulty")]
NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]

# Regroupement des attaques en 4 familles + trafic normal
ATTACK_CATEGORIES = {
    "DoS": ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
            "mailbomb", "processtable", "udpstorm", "worm"],
    "Probe": ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"],
    "R2L": ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
            "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack",
            "snmpguess", "xlock", "xsnoop", "httptunnel"],
    "U2R": ["buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack",
            "xterm"],
}
ATTACK_TO_CLASS = {a: c for c, lst in ATTACK_CATEGORIES.items() for a in lst}
ATTACK_TO_CLASS["normal"] = "normal"

NORMAL_LABEL = {"cls": "normal", "bin": 0}
