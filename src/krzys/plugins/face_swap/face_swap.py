import os
import random
import shutil
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from urllib.request import urlretrieve

import cv2
import discord
import ffmpeg
import insightface
import numpy as np
import onnxruntime
from insightface.model_zoo.inswapper import INSwapper

from krzys import core
from krzys.plugins.face_swap import config
from krzys.plugins.face_swap.exception import SourceFileIsNotImageError, SourceFileDoesNotContainFaceError, \
    TargetImageDoesNotContainFaceError


def pick_best_execution_provider():
    available_providers = onnxruntime.get_available_providers()
    if 'CoreMLExecutionProvider' in available_providers:
        return 'CoreMLExecutionProvider'
    elif 'CUDAExecutionProvider' in available_providers:
        return 'CUDAExecutionProvider'

    return 'CPUExecutionProvider'


execution_provider = pick_best_execution_provider()
print('Using execution provider:', execution_provider)
face_analyzer = insightface.app.FaceAnalysis(
    name='buffalo_l',
    providers=[execution_provider],
    allowed_modules=['detection', 'recognition']
)
face_analyzer.prepare(ctx_id=0, det_thresh=0.30)

inswapper_128_path = os.path.join(os.path.expanduser('~/.insightface'), 'inswapper_128.onnx')
if not os.path.exists(inswapper_128_path):
    print('Downloading inswapper_128.onnx')
    urlretrieve('https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx', inswapper_128_path)

face_swapper: INSwapper = insightface.model_zoo.get_model(
    inswapper_128_path,
    providers=[execution_provider],
)


def is_image(url: str) -> bool:
    return url.split('?')[0].split('.')[-1].lower() in ('jpg', 'jpeg', 'png', 'webp')


def is_video(url: str) -> bool:
    return not is_image(url)


def process_image(source: bytes, target: bytes) -> bytes:
    """
    :raises SourceFileDoesNotContainFaceError:
    :raises TargetImageDoesNotContainFaceError:

    :param source: bytes
    :param target: bytes
    :return: bytes
    """
    source_image = cv2.imdecode(np.asarray(bytearray(source), dtype=np.uint8), cv2.IMREAD_COLOR)
    source_faces = face_analyzer.get(source_image)
    if not source_faces or len(source_faces) == 0:
        raise SourceFileDoesNotContainFaceError()

    source_face = source_faces[0]

    target_image = cv2.imdecode(np.asarray(bytearray(target), dtype=np.uint8), cv2.IMREAD_COLOR)
    target_faces = face_analyzer.get(target_image)
    if not target_faces or len(target_faces) == 0:
        raise TargetImageDoesNotContainFaceError()

    for target_face in target_faces:
        target_image = face_swapper.get(target_image, target_face, source_face, paste_back=True)

    return cv2.imencode('.jpg', target_image)[1].tobytes()


def split_video(source: bytes, session_name: str, fps: int = 24) -> int:
    """

    :param source: bytes
    :param session_name: str
    :param fps: int = 24
    :return: int Number of extracted frames
    """
    in_file = core.config.get_tmp_file(session_name, 'in_file')
    if os.path.exists(in_file):
        os.unlink(in_file)

    with open(in_file, 'wb') as f:
        f.write(source)

    probe = ffmpeg.probe(in_file)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    process = (
        ffmpeg
        .input(in_file)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', r=fps)
        .run_async(pipe_stdout=True, quiet=True)
    )

    extracted_frames = 0
    max_frames = fps * config.max_video_length
    while True:
        frame = process.stdout.read(width * height * 3)
        if not frame:
            break

        if extracted_frames >= max_frames:
            continue

        frame_array = np.frombuffer(frame, dtype=np.uint8)
        frame = frame_array.reshape((height, width, 3))

        max_dimension = 640
        height, width = frame.shape[:2]

        aspect_ratio = width / float(height)

        if width > height:
            new_width = max_dimension
            new_height = int(max_dimension / aspect_ratio)
        else:
            new_height = max_dimension
            new_width = int(max_dimension * aspect_ratio)

        if new_width != width or new_height != height:
            frame = cv2.resize(frame, (new_width, new_height))

        cv2.imwrite(core.config.get_tmp_file(session_name, 'frames', 'frame_%d.jpg' % extracted_frames), frame)
        extracted_frames += 1

    process.wait()
    return extracted_frames


def join_video(session_name: str, fps: int = 24) -> bytes:
    """

    :param session_name: str
    :param fps: int = 24
    :return: bytes
    """
    frames = core.config.get_tmp_file(session_name, 'frames', 'frame_%d.jpg')
    in_file = core.config.get_tmp_file(session_name, 'in_file')
    out_file = core.config.get_tmp_file(session_name, 'output.mp4')

    video = ffmpeg.input(frames, framerate=fps, start_number=0).video
    audio = ffmpeg.input(in_file).audio
    ffmpeg.output(video, audio, out_file, shortest=None).run(quiet=True)


    result_bytes: bytes
    with open(out_file, 'rb') as f:
        result_bytes = f.read()

    return result_bytes


def process_video_multithreaded2_worker(source_face, session_name: str, frame_id: int) -> None:
    current_frame = core.config.get_tmp_file(session_name, 'frames', 'frame_%d.jpg' % frame_id)
    image = cv2.imread(current_frame)

    target_faces = face_analyzer.get(image)
    if not target_faces or len(target_faces) == 0:
        return

    processed_faces = 0
    for target_face in target_faces:
        image = face_swapper.get(image, target_face, source_face, paste_back=True)
        processed_faces += 1
        if processed_faces >= 3:
            break

    cv2.imwrite(current_frame, image)


async def process_video_multithreaded2(i: discord.Interaction, source: bytes, target: bytes, threads: int, fps: int = 24) -> bytes:
    session_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
    session_dir = core.config.get_tmp_file(session_name)
    os.makedirs(core.config.get_tmp_file(session_name, 'frames'), exist_ok=True)

    await i.edit_original_response(content="Wyciągam klatki w %d fps" % fps)
    frame_count = split_video(target, session_name, fps)

    source_image = cv2.imdecode(np.asarray(bytearray(source), dtype=np.uint8), cv2.IMREAD_COLOR)
    source_faces = face_analyzer.get(source_image)
    if not source_faces or len(source_faces) == 0:
        raise SourceFileDoesNotContainFaceError()

    source_face = source_faces[0]

    start_time = time.time()

    await i.edit_original_response(content="Przetwarzam klatki...")
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        queue: Queue[int] = Queue()
        for a in range(frame_count):
            queue.put(a)

        while not queue.empty():
            future = executor.submit(
                process_video_multithreaded2_worker,
                source_face, session_name, queue.get()
            )
            futures.append(future)

        for future in as_completed(futures):
            future.result()

    end_time = time.time()
    print("Time taken: ", end_time - start_time)
    print("FPS: ", frame_count / (end_time - start_time))

    print("Joining video")

    await i.edit_original_response(content="Scalam video, zaraz kurwa będzie")
    result_bytes = join_video(session_name, fps)

    print("Cleaning up")
    shutil.rmtree(session_dir, ignore_errors=True)

    print("Done!")
    return result_bytes
