## Setup
python -m pip install -r requirements.txt

## Testing
python -m py_compile rotation_functions.py RST_specific_functions.py
python -c "import RST_specific_functions as RST; print('import ok')"