import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The project intentionally keeps app.py as the runnable Flask entry point and
# app/ as the ML/DB package. Load app.py explicitly so pytest does not resolve
# the package directory when it sees `import app`.
spec = importlib.util.spec_from_file_location("churnguard_app", ROOT / "app.py")
churnguard_app = importlib.util.module_from_spec(spec)
sys.modules["churnguard_app"] = churnguard_app
spec.loader.exec_module(churnguard_app)
