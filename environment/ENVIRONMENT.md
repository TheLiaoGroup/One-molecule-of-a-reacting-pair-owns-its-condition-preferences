# Environment

The released figures and Source Data were produced, and rebuild byte for byte, with:

- Python 3.13.3 (64-bit) on Windows 10
- the packages and versions of `requirements.txt` (`pip install -r environment/requirements.txt`)
- matplotlib with the released ChemDraw artwork for all four main figures
- the font Arial in regular, bold and italic (the generators stop if one is missing), from these files (`fonts.json`):

  - normal, weight 400: `arial.ttf`, Arial-, Version 6.90 June 24, 2017, SHA-256 `dd985b5dad37d0d1ef5929e176335152841f85f7dd0fcd259889508ff8be981f`
  - normal, weight 700: `arialbd.ttf`, Arial-BoldMT, Version 7.00, SHA-256 `766f06ac8761f82f25d032a220e89438f6064591af9915061f20b949efdedf69`
  - italic, weight 400: `ariali.ttf`, Arial-ItalicMT, Version 7.00, SHA-256 `79a5742b865cf06891d4ae43e42cac0e5169c7b7b06b5b3c24ef8e0966d27b62`

Where two installed files carry the same Arial style, `rebuild_all.py` gives the files above precedence, so the choice does not depend on the order in which font directories are read. Other versions of the packages or fonts can change the bytes of a PDF, SVG or PNG without changing what is drawn; the Source Data workbooks and tables do not depend on fonts. `rebuild_all.py` records the versions it ran with in `_rebuild/REBUILD_REPORT.json`.

The Python guard records Python file access, permits child processes only through this Python executable, and refuses Python network connections. Rebuild outputs and temporary files stay under `_rebuild/`.
