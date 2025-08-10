"""
의약품 정보 음성 대화 시스템
OCR + RAG + 음성 인터페이스를 통합한 메인 프로그램
"""

import os
import time
from ocr_module import OCRModule
from voice_module import VoiceModule
from llm_module import LLMModule

class MedicineAssistant:
    def __init__(self):
        """시스템 초기화"""
        print("="*60)
        print("🏥 의약품 정보 음성 대화 시스템")
        print("="*60)
        print()
        
        # 모듈 초기화
        print("시스템을 초기화하는 중입니다...")
        print("-"*60)
        
        try:
            self.voice = VoiceModule()
            self.ocr = OCRModule()
            self.llm = LLMModule()
            
            print("-"*60)
            print("✅ 모든 모듈이 준비되었습니다!")
            print()
            
        except Exception as e:
            print(f"❌ 초기화 실패: {e}")
            print("\n필요한 사항을 확인해주세요:")
            print("1. OPENAI_API_KEY가 .env 파일에 설정되어 있는지")
            print("2. 필요한 패키지들이 모두 설치되어 있는지")
            print("3. 마이크가 연결되어 있는지")
            raise
        
        # 상태 변수
        self.ocr_context = None
    
    def show_menu(self):
        """메뉴 표시"""
        print("\n" + "="*60)
        print("📋 메뉴")
        print("-"*60)
        print("  S : 음성 대화 시작")
        print("  Q : 프로그램 종료")
        print("="*60)
    
    def handle_medicine_storage(self):
        """약품 저장 처리 (구현 예정)"""
        print("\n🚧 구현 중...")
        self.voice.speak("약품 저장 기능은 현재 구현 중입니다.")
        time.sleep(1)
    
    def handle_medicine_query(self):
        """약품 질의 처리"""
        # 상자 OCR 안내
        ocr_prompt = "만약 약품 상자가 있다면 상자를 보여준 뒤 S 버튼을 눌러주세요. 만약 없다면 그냥 S 버튼을 눌러주세요."
        print(f"\n🔊 {ocr_prompt}")
        self.voice.speak(ocr_prompt)
        
        # 사용자 입력 대기
        user_input = input("\n📷 준비되면 S를 입력하세요: ").strip().upper()
        
        if user_input == 'S':
            # OCR 처리 시도
            image_path = "test/tylenol2.jpg"
            if os.path.exists(image_path):
                print(f"\n📸 이미지 파일 발견: {image_path}")
                ocr_text, _ = self.ocr.extract_text_with_preprocessing(image_path)
                
                if ocr_text:
                    self.ocr_context = self.ocr.format_for_llm(ocr_text)
                    print("✅ 약품 정보가 추출되었습니다.")
                else:
                    print("⚠️ 이미지에서 텍스트를 추출할 수 없었습니다.")
                    self.ocr_context = None
            else:
                print(f"⚠️ {image_path} 파일이 없습니다. OCR을 건너뜁니다.")
                self.ocr_context = None
        
        # 질문 받기
        question_prompt = "무엇을 물어보시겠습니까?"
        print(f"\n🔊 {question_prompt}")
        self.voice.speak(question_prompt)
        
        # 음성으로 질문 받기 (10초)
        print("\n🎤 음성으로 질문해주세요...")
        user_question = self.voice.listen(duration=10)
        
        if not user_question:
            no_input_msg = "질문을 인식할 수 없었습니다. 다시 시도해주세요."
            print(f"❌ {no_input_msg}")
            self.voice.speak(no_input_msg)
            return
        
        print(f"\n❓ 인식된 질문: {user_question}")
        
        # LLM으로 답변 생성
        print("\n🤔 답변을 생성하는 중...")
        answer = self.llm.query(user_question, self.ocr_context)
        
        # 답변 출력 및 음성 재생
        print("\n" + "="*60)
        print("💊 답변:")
        print("-"*60)
        print(answer)
        print("="*60)
        
        # 음성으로 답변
        self.voice.speak(answer)
        
        # OCR 컨텍스트 초기화
        self.ocr_context = None
    
    def start_voice_interaction(self):
        """음성 상호작용 시작"""
        # 인사말
        greeting = "어떤 것을 원하십니까? 1번 약품 저장, 2번 약품 질의"
        print(f"\n🔊 {greeting}")
        self.voice.speak(greeting)
        
        # 사용자 선택 받기 (키보드 입력으로 간소화)
        print("\n선택해주세요:")
        print("  1: 약품 저장")
        print("  2: 약품 질의")
        
        choice = input("\n번호 입력 (1 또는 2): ").strip()
        
        if choice == '1':
            self.handle_medicine_storage()
        elif choice == '2':
            self.handle_medicine_query()
        else:
            error_msg = "잘못된 선택입니다. 다시 시도해주세요."
            print(f"❌ {error_msg}")
            self.voice.speak(error_msg)
    
    def run(self):
        """메인 실행 루프"""
        print("\n🚀 시스템이 준비되었습니다!")
        print("💡 팁: 음성 인식이 잘 되지 않는 경우 조용한 환경에서 시도해주세요.")
        
        while True:
            try:
                self.show_menu()
                command = input("\n명령을 입력하세요: ").strip().upper()
                
                if command == 'Q':
                    print("\n👋 프로그램을 종료합니다.")
                    farewell = "의약품 정보 시스템을 이용해 주셔서 감사합니다. 안녕히 가세요."
                    self.voice.speak(farewell)
                    break
                    
                elif command == 'S':
                    self.start_voice_interaction()
                    
                else:
                    print("❓ 알 수 없는 명령입니다. S 또는 Q를 입력해주세요.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 프로그램을 종료합니다.")
                break
                
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                print("다시 시도해주세요.")
                time.sleep(2)

def check_requirements():
    """필수 요구사항 확인"""
    print("🔍 시스템 요구사항을 확인하는 중...")
    
    requirements = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY") is not None,
        "e_data.json": os.path.exists("e_data.json"),
        "n_data.json": os.path.exists("n_data.json"),
        "test/tylenol2.jpg": os.path.exists("test/tylenol2.jpg")
    }
    
    all_ok = True
    for item, status in requirements.items():
        if status:
            print(f"  ✅ {item}")
        else:
            print(f"  ❌ {item} - {'설정 필요' if 'API' in item else '파일 없음'}")
            all_ok = False
    
    if not all_ok:
        print("\n⚠️ 일부 요구사항이 충족되지 않았습니다.")
        print("다음 사항을 확인해주세요:")
        
        if not requirements["OPENAI_API_KEY"]:
            print("1. .env 파일에 OPENAI_API_KEY를 설정하세요")
            
        if not requirements["e_data.json"] or not requirements["n_data.json"]:
            print("2. 의약품 데이터 파일(e_data.json, n_data.json)을 준비하세요")
            print("   (없어도 실행은 가능하지만 RAG 기능이 제한됩니다)")
            
        if not requirements["test/tylenol2.jpg"]:
            print("3. OCR 테스트용 이미지 파일(test/tylenol2.jpg)을 준비하세요")
            print("   (없어도 실행은 가능하지만 OCR 기능을 사용할 수 없습니다)")
        
        print("\n계속 진행하시겠습니까? (y/n)")
        if input().strip().lower() != 'y':
            return False
    
    return True

def main():
    """메인 함수"""
    print("\n" + "="*60)
    print("🏥 의약품 정보 음성 대화 시스템 v1.0")
    print("="*60)
    print()
    print("이 시스템은 다음 기능을 제공합니다:")
    print("• OCR을 통한 약품 상자 텍스트 인식")
    print("• RAG 기반 의약품 정보 검색")
    print("• 음성 인터페이스 (STT/TTS)")
    print()
    
    # 요구사항 확인
    if not check_requirements():
        print("프로그램을 종료합니다.")
        return
    
    print()
    
    try:
        # 시스템 시작
        assistant = MedicineAssistant()
        assistant.run()
        
    except Exception as e:
        print(f"\n❌ 치명적 오류 발생: {e}")
        print("\n문제 해결 방법:")
        print("1. 필요한 패키지 설치 확인:")
        print("   pip install paddleocr faster-whisper gtts pygame pyaudio")
        print("   pip install llama-index llama-index-llms-openai llama-index-embeddings-openai")
        print("   pip install python-dotenv opencv-python numpy")
        print()
        print("2. 마이크 연결 상태 확인")
        print("3. 인터넷 연결 상태 확인 (gTTS, OpenAI API 사용)")
        print("4. .env 파일에 OPENAI_API_KEY 설정 확인")
        
        import traceback
        print("\n상세 오류 정보:")
        traceback.print_exc()

if __name__ == "__main__":
    main()