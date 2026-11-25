import sys
from pathlib import Path

# make `import eegad` work without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
