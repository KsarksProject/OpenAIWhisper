import os
import warnings
import time
from flask import Flask, request, render_template, jsonify
import ffmpeg
import whisper

# Подавляем предупреждения
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Flask-приложение
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = "cuda" if whisper.torch.cuda.is_available() else "cpu"
model = whisper.load_model("small", device=device)


def get_video_metadata(video_path):
    """Получение информации о видеофайле с помощью ffmpeg."""
    try:
        probe = ffmpeg.probe(video_path)
        video_info = {
            "duration": float(probe["format"]["duration"]),  # Длительность видео в секундах
            "size": int(probe["format"]["size"]),  # Размер файла в байтах
            "bit_rate": int(probe["format"]["bit_rate"]),  # Битрейт
            "format_name": probe["format"]["format_name"],  # Формат файла
        }
        return video_info
    except Exception as e:
        return {"error": str(e)}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file.filename == "":
            return "Ошибка: имя файла пустое", 400

        # Сохраняем видеофайл
        video_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(video_path)

        # Получаем информацию о видео
        video_metadata = get_video_metadata(video_path)
        if "error" in video_metadata:
            return f"Ошибка анализа видео: {video_metadata['error']}", 500

        # Конвертируем видео в аудио (WAV)
        audio_path = video_path.rsplit(".", 1)[0] + ".wav"
        ffmpeg.input(video_path).output(audio_path, ar=16000, ac=1).run(overwrite_output=True)

        # Транскрибация речи с замером времени
        start_time = time.time()
        result = model.transcribe(audio_path, language="ru", fp16=(device == "cuda"))
        end_time = time.time()

        transcribed_text = result["text"]
        elapsed_time = end_time - start_time  # Время выполнения

        # Возвращаем текст и метаданные
        return jsonify({
            "text": transcribed_text,
            "time": elapsed_time,
            "video_metadata": {
                "duration": round(video_metadata["duration"], 2),  # Длительность в секундах
                "size_mb": round(video_metadata["size"] / (1024 * 1024), 2),  # Размер в МБ
                "bit_rate_kbps": round(video_metadata["bit_rate"] / 1000, 2),  # Битрейт в Кбит/с
                "format_name": video_metadata["format_name"],  # Формат файла
            }
        })

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
