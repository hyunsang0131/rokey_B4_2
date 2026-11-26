"""시스템 파이썬에 없는 패키지(mediapipe, anthropic 등)를 별도 venv 에서 찾아 import.

ros2 run 스크립트는 /usr/bin/python3 로 돌기 때문에, 없으면 ASSIST_MP_VENV(기본 ~/venvs/assist_mp)의
site-packages 를 sys.path *뒤쪽*에 붙여 다시 시도한다 (시스템 numpy/cv2/rclpy 가 우선).
"""
import glob
import importlib
import os
import sys


def venv_site_packages():
    venv = os.path.expanduser(os.environ.get('ASSIST_MP_VENV', '~/venvs/assist_mp'))
    return sorted(glob.glob(os.path.join(venv, 'lib', 'python3*', 'site-packages')))


def import_from_venv(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    for sp in venv_site_packages():
        if sp not in sys.path:
            sys.path.append(sp)
    return importlib.import_module(name)   # 실패하면 ImportError 그대로
