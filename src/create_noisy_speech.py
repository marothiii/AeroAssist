from pathlib import Path
import subprocess


PROJECT_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_DIR / "audio"

FILES = [
    "speech_01.m4a",
    "speech_02.m4a",
    "speech_03.m4a",
    "speech_04.m4a",
    "speech_05.m4a",
]


def main():

    print("Creating noisy speech samples...\n")

    for filename in FILES:
        input_path = AUDIO_DIR / filename

        if not input_path.exists():
            raise FileNotFoundError(
                f"Missing audio file: {input_path}"
            )

        output_name = filename.replace(
            ".m4a",
            "_noisy.m4a",
        )

        output_path = AUDIO_DIR / output_name

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),

            "-filter_complex",
            (
                "anoisesrc=color=pink:"
                "amplitude=0.08"
                "[noise];"
                "[0:a][noise]"
                "amix=inputs=2:"
                "duration=first:"
                "weights='1 0.35'"
            ),

            "-c:a",
            "aac",

            str(output_path),
        ]

        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(
            f"Created: {output_name}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()