import cv2
import numpy as np
import threading
import queue
import time


class VideoCaptureThread(threading.Thread):
    """Поток для захвата кадров с камеры"""

    def __init__(self, src=0, buffer_size=2):
        super().__init__()
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise ValueError("Ошибка открытия камеры")
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = True
        self.daemon = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame.copy())
        self.cap.release()

    def get_frame(self, timeout=0.5):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        time.sleep(0.1)
        while not self.frame_queue.empty():
            self.frame_queue.get()


def apply_sharpening(frame):
    """Применение фильтра резкости"""
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(frame, -1, kernel)


def detect_edges(frame):
    """Выделение контуров"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Конвертируем обратно в BGR для отображения


def main():
    # Инициализация видеопотока
    capture_thread = VideoCaptureThread()
    capture_thread.start()

    while True:
        frame = capture_thread.get_frame()
        if frame is None:
            continue

        # 1. Оригинальный кадр
        original = frame.copy()

        # 2. Фильтр резкости
        sharpened = apply_sharpening(frame)

        # 3. Выделение контуров
        edges = detect_edges(frame)

        # Масштабируем кадры для отображения
        scale_percent = 60  # Размер для трех окон
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        original_resized = cv2.resize(original, dim)
        sharpened_resized = cv2.resize(sharpened, dim)
        edges_resized = cv2.resize(edges, dim)

        # Добавляем подписи
        cv2.putText(original_resized, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(sharpened_resized, "Sharpened", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(edges_resized, "Edges", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(edges_resized, "Press 'E' to quit", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Объединяем все три кадра в одно окно
        combined = np.hstack((original_resized, sharpened_resized, edges_resized))

        cv2.imshow("Video Processing: Original | Sharpened | Edges", combined)

        # Завершение работы по нажатию 'E'
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    # Корректное завершение
    capture_thread.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
