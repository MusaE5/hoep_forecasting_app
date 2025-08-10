import subprocess
import os

def main(request):
    """Cloud Function entry point for HOEP live prediction."""
    script_path = os.path.join(os.path.dirname(__file__), "live_prediction.py")

    result = subprocess.run(
        ["python", script_path],
        capture_output=True,
        text=True
    )

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    return "HOEP forecast job complete."
