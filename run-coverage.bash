#!/usr/bin/env bash

pytest --cov=leftcorner tests/tests.py
coverage html
xdg-open htmlcov/index.html
