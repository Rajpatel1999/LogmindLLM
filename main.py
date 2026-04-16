import sys

try:
    import torch  # noqa: F401
except ImportError:
    print(
        "PyTorch is not installed for this Python interpreter:\n"
        f"  {sys.executable}\n\n"
        "From the project root, use the bundled script (creates .venv if needed):\n"
        "  sh run.sh\n\n"
        "Or activate the venv yourself:\n"
        "  python3 -m venv .venv\n"
        "  .venv/bin/pip install -r requirements.txt\n"
        "  .venv/bin/python main.py",
        file=sys.stderr,
    )
    raise SystemExit(1) from None

from utils.tokenizer import CharTokenizer
from data.dataset import TextDataset

text = "API failed request API timeout error"

tokenizer = CharTokenizer()
tokenizer.build_vocab(text)

dataset = TextDataset(text, tokenizer, seq_length=5)

input_tensor, target_tensor = dataset[0]

print("Input tensor:", input_tensor)
print("Target tensor:", target_tensor)
