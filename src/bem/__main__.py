"""__main__.py — package entry point.

Allows the package to be invoked as:
    python -m bem --config configs/run_config.yaml
"""

from bem.run import main

if __name__ == "__main__":
    main()
