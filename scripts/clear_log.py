import os
log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'app.log')
with open(log_path, 'w') as f:
    f.write('')