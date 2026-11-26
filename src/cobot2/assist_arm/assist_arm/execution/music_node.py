"""음악 추천·재생 (로컬 파일만). 외부 스트리밍/추천 API 는 인터페이스(MusicProvider)만 두고 호출하지 않는다.

입력  /music/play  (std_msgs/String JSON {"mood": "신나는"|"any", "title": 선택})
      /music/stop  (std_msgs/Empty)
출력  /music/now_playing (std_msgs/String JSON {"playing": bool, "title", "file", "bpm", "mood", "duration_sec"})
재생기: ffplay / paplay / aplay / mpg123 중 시스템에 *이미 있는* 것을 사용 (설치하지 않음). 없으면 재생 없이
        now_playing 만 발행 → 율동은 bpm 으로 그대로 동작 (데모 때 스피커는 따로 틀어도 됨).
곡 목록: config/music_library.yaml (title, file, bpm, moods, duration_sec)
"""
import json
import os
import random
import shutil
import subprocess
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Empty, String

from assist_arm.common.heartbeat import attach_heartbeat

PLAYERS = [
    ('ffplay', ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet']),
    ('paplay', ['paplay']),
    ('aplay', ['aplay', '-q']),
    ('mpg123', ['mpg123', '-q']),
]


class MusicProvider:
    """추천 인터페이스. recommend(mood) → track dict | None"""

    def recommend(self, mood):
        raise NotImplementedError


class LocalLibraryProvider(MusicProvider):
    def __init__(self, tracks, base_dir):
        self.tracks, self.base = tracks, base_dir

    def recommend(self, mood):
        cand = [t for t in self.tracks if mood in ('any', '') or mood in t.get('moods', [])]
        if not cand:
            cand = self.tracks
        if not cand:
            return None
        t = dict(random.choice(cand))
        t['file'] = os.path.expanduser(os.path.join(self.base, t.get('file', '')))
        return t


class ExternalApiProvider(MusicProvider):
    """자리표시: 외부 음악 API (Spotify/YouTube 등). 이 프로젝트에서는 호출하지 않는다."""

    def recommend(self, mood):
        raise NotImplementedError('외부 API 연동은 범위 밖 — LocalLibraryProvider 사용')


class MusicNode(Node):
    def __init__(self):
        super().__init__('music_node')
        self.declare_parameter('library_file', '')
        self.declare_parameter('music_dir', '~/data/assist_arm/music')
        lib = self.get_parameter('library_file').value or os.path.join(
            get_package_share_directory('assist_arm'), 'config', 'music_library.yaml')
        with open(os.path.expanduser(lib)) as f:
            data = yaml.safe_load(f) or {}
        self.provider = LocalLibraryProvider(data.get('tracks', []), self.get_parameter('music_dir').value)
        self.player = next((cmd for name, cmd in PLAYERS if shutil.which(name)), None)
        if self.player is None:
            self.get_logger().warn('오디오 재생기(ffplay/paplay/aplay/mpg123) 없음 → now_playing 만 발행')
        self.proc = None
        self.cur = None
        self.t_start = 0.0
        self.pub = self.create_publisher(String, '/music/now_playing', 10)
        self.create_subscription(String, '/music/play', self.on_play, 10)
        self.create_subscription(Empty, '/music/stop', lambda _: self.stop(), 10)
        self.create_timer(0.5, self.tick)
        attach_heartbeat(self)

    def on_play(self, msg):
        try:
            req = json.loads(msg.data or '{}')
        except ValueError:
            req = {}
        self.stop()
        t = self.provider.recommend(req.get('mood', 'any'))
        if t is None:
            self.get_logger().error('music_library.yaml 에 곡이 없습니다')
            return
        t['mood'] = req.get('mood', 'any')
        self.cur, self.t_start = t, time.monotonic()
        if self.player and os.path.exists(t['file']):
            self.proc = subprocess.Popen(self.player + [t['file']], stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
        elif self.player:
            self.get_logger().warn(f"파일 없음: {t['file']} (재생 없이 bpm 만 사용)")
        self.get_logger().info(f"추천/재생: {t.get('title')} ({t.get('bpm')} bpm, mood={t['mood']})")
        self.publish(True)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None
        if self.cur:
            self.publish(False)
        self.cur = None

    def tick(self):
        if not self.cur:
            return
        ended = (self.proc is not None and self.proc.poll() is not None) or (
            self.proc is None and time.monotonic() - self.t_start > float(self.cur.get('duration_sec', 30)))
        if ended:
            self.stop()
        else:
            self.publish(True)

    def publish(self, playing):
        d = dict(self.cur or {})
        d['playing'] = playing
        self.pub.publish(String(data=json.dumps(d, ensure_ascii=False)))

    def destroy_node(self):
        self.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MusicNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
