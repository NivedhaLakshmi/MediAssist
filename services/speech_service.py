from faster_whisper import WhisperModel
import os


# Load model once when the service starts

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)


def transcribe_audio(audio_file):
    """
    Convert recorded audio into text.
    """

    print("================================")
    print("Speech-to-Text Debug")
    print("Audio file:", audio_file)
    print("Exists:", os.path.exists(audio_file))

    if os.path.exists(audio_file):

        print(
            "File size:",
            os.path.getsize(audio_file),
            "bytes"
        )

    print("================================")


    segments, info = model.transcribe(
        audio_file,
        beam_size=5
    )


    print(
        "Detected language:",
        info.language
    )

    print(
        "Language probability:",
        info.language_probability
    )


    text_parts = []


    for segment in segments:

        print(
            f"[{segment.start:.2f} -> "
            f"{segment.end:.2f}] "
            f"{segment.text}"
        )

        text_parts.append(
            segment.text.strip()
        )


    text = " ".join(
        text_parts
    ).strip()


    print(
        "Final transcription:",
        repr(text)
    )


    return text