"""Installation script for Playwright browsers."""

import subprocess
import sys


def install_browsers():
    """Install Playwright browsers."""
    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install"],
            check=True,
            capture_output=True,
        )
        print("✅ Playwright browsers installed successfully!")
    except subprocess.CalledProcessError as e:
        print("❌ Failed to install Playwright browsers!")
        print(e.stderr.decode())
        sys.exit(1)


if __name__ == "__main__":
    install_browsers()
