"""Allow running as: python -m ToyOption"""
from .app import app

app.run(debug=True, port=8050)
