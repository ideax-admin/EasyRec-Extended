#!/bin/bash
# Start the serving service
export FLASK_APP=app.py
export ENV=${ENV:-production}
gunicorn -w 4 -b 0.0.0.0:5000 app:app
