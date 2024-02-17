from setuptools import setup, find_namespace_packages

setup(
    name='krzys-plugins-face_swap',
    version='1.0.0',
    author="iTokajo",
    packages=find_namespace_packages(where='src/', include=['krzys.plugins.face_swap']),
    package_dir={
        '': 'src'
    },
    entry_points={
        'krzys.plugins': [
            'face_swap = krzys.plugins.face_swap:Plugin',
        ]
    },
    dependency_links=[
        'https://download.pytorch.org/whl/cu118',
    ],
    install_requires=[
        'ffmpeg-python',
        'numpy',
        'opencv-python',
        'onnx',
        'insightface',
        'onnxruntime; python_version != "3.9" and sys_platform == "darwin" and platform_machine != "arm64"',
        'onnxruntime-coreml; python_version == "3.9" and sys_platform == "darwin" and platform_machine != "arm64"',
        'onnxruntime-silicon; sys_platform == "darwin" and platform_machine == "arm64"',
        'onnxruntime-gpu==1.15.1; sys_platform != "darwin"',
        'tensorflow',
    ],
)
