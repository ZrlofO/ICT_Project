"""
음성 처리 모듈 - STT와 TTS 기능 제공
Windows 파일 잠금 문제 해결 버전
"""

import pyaudio
import wave
import tempfile
import os
import time
import numpy as np
from faster_whisper import WhisperModel
from gtts import gTTS
import pygame
import gc

class VoiceModule:
    def __init__(self):
        """음성 모듈 초기화"""
        print("🎤 음성 모듈을 초기화하는 중...")
        
        # Whisper 모델 로드
        print("  📥 Whisper 모델 로드 중...")
        self.whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
        
        # 오디오 설정
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        
        # PyAudio 초기화
        self.audio = pyaudio.PyAudio()
        
        # Pygame 초기화 (TTS 재생용)
        pygame.mixer.init()
        
        # 임시 파일 추적
        self.temp_files = []
        
        print("✅ 음성 모듈 준비 완료!")
    
    def _cleanup_temp_files(self):
        """임시 파일 정리"""
        for filepath in self.temp_files[:]:
            try:
                if os.path.exists(filepath):
                    os.unlink(filepath)
                self.temp_files.remove(filepath)
            except:
                pass  # 삭제 실패해도 무시
    
    def speak(self, text, speed_mode='normal'):
        """
        텍스트를 음성으로 변환하여 재생 (속도 조절 가능)
        
        Args:
            text: 음성으로 변환할 텍스트
            speed_mode: 'slow', 'normal', 'fast' 중 선택
        """
        if not text or not text.strip():
            return
        
        print(f"🔊 음성 출력 ({speed_mode}): '{text[:50]}...'")
        
        temp_file = None
        try:
            # 속도 모드에 따른 설정
            if speed_mode == 'slow':
                # 방법 1: gTTS의 slow 옵션 사용
                tts = gTTS(text=text, lang='ko', slow=True)
                
            elif speed_mode == 'fast':
                # 빠른 재생을 위한 설정
                tts = gTTS(text=text, lang='ko', slow=False)
                
            else:  # normal
                tts = gTTS(text=text, lang='ko', slow=False)
            
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                temp_file = tmp.name
                self.temp_files.append(temp_file)
            
            # TTS 저장
            tts.save(temp_file)
            
            # 속도별 재생 설정
            if speed_mode == 'fast':
                # 1.25배속 재생
                pygame.mixer.quit()
                pygame.mixer.init(frequency=27562)  # 22050 * 1.25
            elif speed_mode == 'slow':
                # 이미 gTTS slow 옵션 적용됨
                pygame.mixer.quit()
                pygame.mixer.init(frequency=22050)
            else:
                # 일반 속도
                pygame.mixer.quit()
                pygame.mixer.init(frequency=22050)
            
            # 재생
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # 재생 완료 대기 (스킵 기능 포함)
            print("🎵 음성 재생 중... (S키를 누르면 스킵)")
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                # Windows에서 키 입력 체크 (non-blocking)
                try:
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8', errors='ignore').upper()
                        if key == 'S':
                            print("\n⏭️ 음성을 스킵했습니다.")
                            pygame.mixer.music.stop()
                            break
                except ImportError:
                    # Windows가 아닌 경우는 스킵 기능 없이 진행
                    pass
                except:
                    # 기타 예외 발생 시 무시하고 계속
                    pass
            
            # 리소스 해제
            pygame.mixer.music.unload()
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ TTS 오류: {e}")
        finally:
            # 파일 삭제 시도
            if temp_file:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                    if temp_file in self.temp_files:
                        self.temp_files.remove(temp_file)
                except:
                    pass
    
    def listen(self, duration=10):
        """
        마이크에서 음성을 듣고 텍스트로 변환
        
        Args:
            duration: 녹음 시간 (초)
            
        Returns:
            인식된 텍스트
        """
        print(f"🎤 {duration}초간 음성을 듣고 있습니다... 말씀해주세요!")
        
        stream = None
        temp_file = None
        
        try:
            # 녹음 스트림 열기
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            frames = []
            start_time = time.time()
            
            # 녹음
            while time.time() - start_time < duration:
                remaining = duration - int(time.time() - start_time)
                print(f"\r⏱️ 남은 시간: {remaining}초  ", end="", flush=True)
                
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    frames.append(data)
                    
                    # 볼륨 체크
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    volume = np.abs(audio_data).mean()
                    if volume > 500:
                        print("🔊", end="", flush=True)
                except:
                    continue
            
            print(f"\n✅ 녹음 완료!")
            
            # 스트림 정리
            if stream:
                stream.stop_stream()
                stream.close()
                stream = None
            
            # 오디오 데이터를 파일로 저장
            audio_data = b''.join(frames)
            
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_file = tmp.name
                self.temp_files.append(temp_file)
            
            # WAV 파일 작성
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(audio_data)
            
            # STT 처리
            text = self._speech_to_text(temp_file)
            
            return text
            
        except KeyboardInterrupt:
            print("\n⏹️ 녹음이 중단되었습니다.")
            return None
        except Exception as e:
            print(f"\n❌ 녹음 오류: {e}")
            return None
        finally:
            # 스트림 정리
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
            
            # 파일 삭제 시도
            if temp_file:
                try:
                    time.sleep(0.5)  # Windows 파일 시스템 대기
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                    if temp_file in self.temp_files:
                        self.temp_files.remove(temp_file)
                except:
                    pass  # 삭제 실패해도 계속 진행
    
    def _speech_to_text(self, audio_file_path):
        """
        음성 파일을 텍스트로 변환
        
        Args:
            audio_file_path: 오디오 파일 경로
            
        Returns:
            인식된 텍스트
        """
        if not os.path.exists(audio_file_path):
            print("❌ 오디오 파일을 찾을 수 없습니다.")
            return None
        
        print("🔄 음성을 텍스트로 변환하는 중...")
        
        try:
            # Whisper로 변환
            segments, info = self.whisper_model.transcribe(
                audio_file_path,
                language="ko",
                beam_size=5,
                best_of=5,
                temperature=0.0
            )
            
            # 결과 텍스트 조합
            transcribed_text = ""
            for segment in segments:
                transcribed_text += segment.text
            
            transcribed_text = transcribed_text.strip()
            
            if transcribed_text:
                print(f"📝 인식된 텍스트: '{transcribed_text}'")
                return transcribed_text
            else:
                print("❌ 음성을 인식할 수 없습니다.")
                return None
                
        except Exception as e:
            print(f"❌ STT 오류: {e}")
            return None
    
    def __del__(self):
        """정리 작업"""
        # 남은 임시 파일 모두 삭제
        self._cleanup_temp_files()
        
        # PyAudio 정리
        if hasattr(self, 'audio'):
            try:
                self.audio.terminate()
            except:
                pass
        
        # Pygame 정리
        try:
            pygame.mixer.quit()
        except:
            pass