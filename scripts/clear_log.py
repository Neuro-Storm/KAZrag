from pathlib import Path

log_path = Path(__file__).parent.parent / 'logs' / 'app.log'
log_path.parent.mkdir(parents=True, exist_ok=True)
log_path.write_text('')