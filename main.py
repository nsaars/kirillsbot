from dotenv import load_dotenv

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

load_dotenv()

AUDIO_FILE = "/home/malik/PycharmProjects/kirillsbot/app/utils/ai_assistant/audio_2024-09-09_21-07-14.ogg"

API_KEY = "1840a8b0de926f8e1b7729d814cc015d034c9678"


def main():
    try:
        deepgram = DeepgramClient(API_KEY)

        with open(AUDIO_FILE, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            language='ru',
            numerals=True,

        )

        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

        print(response.to_json(indent=4).encode('utf-8').decode('unicode-escape'))

    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    main()

