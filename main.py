"""
의약품 정보 음성 대화 시스템
OCR + RAG + 음성 인터페이스를 통합한 메인 프로그램
"""

import os
import time
import pandas as pd
import cv2
from modules.ocr_module import OCRModule
from modules.voice_module import VoiceModule
from modules.llm_module import LLMModule
from modules.store import StoreModule

class MedicineAssistant:
    def __init__(self):
        """시스템 초기화 (모든 것을 미리 로드)"""
        print("="*60)
        print("🏥 의약품 정보 음성 대화 시스템")
        print("="*60)
        print()
        
        print("시스템을 초기화하는 중입니다...")
        print("🚀 빠른 사용자 경험을 위해 모든 모듈을 미리 준비합니다...")
        print("-"*60)
        
        try:
            # 모든 모듈 미리 로드
            print("🎤 음성 모듈 로드 중...")
            self.voice = VoiceModule()
            print("✅ 음성 모듈 완료")
            
            print("📷 OCR 모듈 로드 중...")
            self.ocr = OCRModule()
            print("✅ OCR 모듈 완료")
            
            print("🤖 LLM 모듈 로드 중...")
            self.llm = LLMModule()
            print("✅ LLM 모듈 완료")
            
            print("💾 저장 모듈 로드 중...")
            self.store = StoreModule(self.voice, self.ocr, self.llm)
            print("✅ 저장 모듈 완료")
            
        except Exception as e:
            print(f"❌ 초기화 실패: {e}")
            print("\n필요한 사항을 확인해주세요:")
            print("1. OPENAI_API_KEY가 .env 파일에 설정되어 있는지")
            print("2. 필요한 패키지들이 모두 설치되어 있는지")
            print("3. 마이크가 연결되어 있는지")
            raise
        
        # 상태 변수
        self.ocr_context = None
        
        # 의약품 이미지 경로 (카메라로 촬영됨)
        self.med_img = "scan/capture.jpg"

        # scan 폴더 생성
        os.makedirs("scan", exist_ok=True)

        # 카메라 설정
        self.camera_index = 1  # Camo 카메라 번호
        
        # CSV 캐시
        self.csv_cache = {
            'general': None,
            'given': None,
            'last_modified': {}
        }
        
        # 검색 인덱스 (빠른 매칭을 위해)
        self.search_index = []
        
        # CSV 데이터 및 검색 인덱스 로드
        try:
            print("📊 약품 데이터 로드 중...")
            self.load_csv_data()
            self.build_search_index()
            print("✅ 약품 데이터 완료")
            
            print("-"*60)
            print("🎯 모든 준비가 완료되었습니다! 빠른 반응 준비됨!")
            print()
        except Exception as e:
            print(f"⚠️ 약품 데이터 로드 중 오류 발생: {e}")
            print("시스템은 계속 실행되지만 일부 기능이 제한될 수 있습니다.")
    
    def load_csv_data(self):
        """CSV 파일을 로드하고 메모리에 저장"""
        csv_files = ['user_med_data/general.csv', 'user_med_data/given.csv']
        
        for csv_name in csv_files:
            try:
                if os.path.exists(csv_name):
                    cache_key = csv_name.split('.')[0]  # 'general' or 'given'
                    self.csv_cache[cache_key] = pd.read_csv(csv_name, encoding='utf-8')
                    self.csv_cache['last_modified'][cache_key] = os.path.getmtime(csv_name)
                else:
                    cache_key = csv_name.split('.')[0]
                    self.csv_cache[cache_key] = pd.DataFrame()
                    
            except Exception as e:
                print(f"⚠️ {csv_name} 로드 실패: {e}")
                cache_key = csv_name.split('.')[0]
                self.csv_cache[cache_key] = pd.DataFrame()
    
    def build_search_index(self):
        """빠른 검색을 위한 인덱스 구축"""
        self.search_index = []
        
        # general.csv 인덱싱
        general_df = self.csv_cache.get('general')
        if general_df is not None and not general_df.empty:
            for idx, row in general_df.iterrows():
                if '약품명' in row and pd.notna(row['약품명']):
                    medicine_name = str(row['약품명'])
                    self.search_index.append({
                        'name': medicine_name,
                        'normalized': medicine_name.lower().replace(" ", "").replace("-", ""),
                        'keywords': medicine_name.lower().split(),
                        'row': idx + 2,
                        'source': 'general'
                    })
        
        # given.csv 인덱싱
        given_df = self.csv_cache.get('given')
        if given_df is not None and not given_df.empty:
            for idx, row in given_df.iterrows():
                if '제목' in row and pd.notna(row['제목']):
                    medicine_name = str(row['제목'])
                    self.search_index.append({
                        'name': medicine_name,
                        'normalized': medicine_name.lower().replace(" ", "").replace("-", ""),
                        'keywords': medicine_name.lower().split(),
                        'row': idx + 2,
                        'source': 'given'
                    })
        
        print(f"📚 검색 인덱스 구축 완료: {len(self.search_index)}개 약품")
    
    def fast_search_medicine(self, user_input):
        """미리 구축된 인덱스를 사용한 고속 약품 검색"""
        if not self.search_index:
            return None, []
        
        user_normalized = user_input.lower().replace(" ", "").replace("-", "")
        user_keywords = user_input.lower().split()
        
        exact_matches = []
        partial_matches = []
        keyword_matches = []
        
        for item in self.search_index:
            # 정확한 매치 (정규화된 문자열)
            if (user_normalized in item['normalized'] or 
                item['normalized'] in user_normalized):
                if len(user_normalized) > 2:
                    exact_matches.append(item)
                    continue
            
            # 키워드 매치
            keyword_score = 0
            for user_word in user_keywords:
                if len(user_word) > 1:
                    for med_word in item['keywords']:
                        if user_word in med_word or med_word in user_word:
                            keyword_score += 1
                            break
            
            if keyword_score > 0:
                item_with_score = item.copy()
                item_with_score['score'] = keyword_score
                keyword_matches.append(item_with_score)
        
        # 정확한 매치가 있으면 첫 번째 반환
        if exact_matches:
            return exact_matches[0], []
        
        # 키워드 점수로 정렬
        keyword_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # 최고 점수가 여러 개면 부분 매치로, 하나면 정확한 매치로 처리
        if keyword_matches:
            best_score = keyword_matches[0]['score']
            best_matches = [m for m in keyword_matches if m['score'] == best_score]
            
            if len(best_matches) == 1:
                return best_matches[0], []
            else:
                return None, best_matches[:5]  # 최대 5개까지
        
        return None, []
    
    def extract_medicine_from_sentence(self, user_input):
        """GPT를 사용한 자연어에서 약품명 추출"""
        
        # 저장된 약품 목록을 간단하게 정리
        medicine_list = [item['name'] for item in self.search_index]
        
        # 약품이 너무 많으면 상위 몇 개만 GPT에 보내기 (토큰 절약)
        if len(medicine_list) > 20:
            # 일단 간단한 키워드 매칭으로 후보를 줄임
            user_lower = user_input.lower()
            potential_medicines = []
            
            for item in self.search_index:
                medicine_name = item['name'].lower()
                words = medicine_name.split()
                
                # 사용자 입력과 겹치는 단어가 있는지 확인
                for word in words:
                    if len(word) > 2 and word in user_lower:
                        potential_medicines.append(item)
                        break
                
                # 사용자 입력의 단어가 약품명에 있는지 확인  
                user_words = user_lower.split()
                for user_word in user_words:
                    if len(user_word) > 2 and user_word in medicine_name:
                        if item not in potential_medicines:
                            potential_medicines.append(item)
                        break
            
            if potential_medicines:
                medicine_list = [item['name'] for item in potential_medicines[:15]]
            else:
                medicine_list = medicine_list[:15]  # 아무 매치도 없으면 처음 15개
        
        # GPT로 약품명 추출
        extract_prompt = f"""사용자가 말한 내용: "{user_input}"

사용 가능한 약품 목록:
{chr(10).join(f'- {med}' for med in medicine_list)}

사용자가 말한 내용을 분석해서 어떤 약품을 원하는지 파악해주세요.

예시:
- "타이레놀인가 그거 꺼내줘" → 타이레놀 8시간 이알서방정 650 mg
- "두통약 좀 가져다줘" → (두통에 효과있는 약품명)  
- "그 진통제 있잖아" → (진통제 약품명)
- "감기약 좀" → (감기약 약품명)

위 약품 목록에서 사용자가 원하는 약품의 정확한 이름을 하나만 출력하세요.
일치하는 약품이 없으면 "없음"이라고 답하세요.
약품명만 출력하고 다른 설명은 하지 마세요."""

        try:
            extracted_name = self.llm.query(extract_prompt, None).strip()
            
            if extracted_name == "없음" or not extracted_name:
                return None, []
            
            # 추출된 약품명을 인덱스에서 찾기
            for item in self.search_index:
                if (extracted_name.lower() in item['name'].lower() or 
                    item['name'].lower() in extracted_name.lower()):
                    return item, []
            
            # 정확한 매치가 없으면 부분 매치 찾기
            partial_matches = []
            extracted_words = extracted_name.lower().split()
            
            for item in self.search_index:
                item_words = item['name'].lower().split()
                match_score = 0
                
                for ext_word in extracted_words:
                    if len(ext_word) > 2:
                        for item_word in item_words:
                            if ext_word in item_word or item_word in ext_word:
                                match_score += 1
                                break
                
                if match_score > 0:
                    partial_matches.append({
                        'item': item,
                        'score': match_score
                    })
            
            if partial_matches:
                # 점수가 가장 높은 것들만 반환
                partial_matches.sort(key=lambda x: x['score'], reverse=True)
                best_score = partial_matches[0]['score']
                best_matches = [m['item'] for m in partial_matches if m['score'] == best_score]
                
                if len(best_matches) == 1:
                    return best_matches[0], []
                else:
                    return None, best_matches[:3]
            
        except Exception as e:
            print(f"⚠️ GPT 추출 오류: {e}")
        
        return None, []
    
    def show_menu(self):
        """메뉴 표시"""
        print("\n" + "="*60)
        print("📋 메뉴")
        print("-"*60)
        print("  S : 음성 대화 시작")
        print("  Q : 프로그램 종료")
        print("="*60)
    
    def capture_medicine_image(self):
        """카메라로 의약품 이미지 촬영"""
        print("\n📷 카메라를 준비하는 중...")

        try:
            # CAP_DSHOW 백엔드로 카메라 열기
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            time.sleep(2)  # 카메라 초기화 대기

            if not cap.isOpened():
                print(f"❌ 카메라 {self.camera_index}번을 열 수 없습니다.")
                # 다른 카메라 인덱스 시도
                for i in range(3):
                    if i != self.camera_index:
                        print(f"카메라 {i}번으로 재시도...")
                        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                        time.sleep(1)
                        if cap.isOpened():
                            self.camera_index = i
                            print(f"✅ 카메라 {i}번 연결 성공")
                            break
                else:
                    raise Exception("사용 가능한 카메라를 찾을 수 없습니다.")

            print(f"✅ 카메라 {self.camera_index}번에 연결되었습니다.")

            # 프레임 촬영
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(self.med_img, frame)
                print(f"📸 사진이 {self.med_img}로 저장되었습니다.")
                success = True
            else:
                print("❌ 카메라에서 이미지를 읽을 수 없습니다.")
                success = False

            # 카메라 리소스 해제
            cap.release()
            return success

        except Exception as e:
            print(f"❌ 카메라 오류: {e}")
            return False

    def handle_medicine_storage(self):
        """약품 저장 처리"""
        print("\n💾 약품 저장 프로세스를 시작합니다...")

        # 카메라로 사진 촬영
        if not self.capture_medicine_image():
            fail_msg = "카메라 촬영에 실패했습니다."
            print(f"\n❌ {fail_msg}")
            self.voice.speak(fail_msg)
            return

        try:
            result = self.store.start_storage_process(self.med_img)
            if result:
                success_msg = "약품 저장이 완료되었습니다."
                print(f"\n✅ {success_msg}")
                self.voice.speak(success_msg)
            else:
                fail_msg = "약품 저장이 취소되었습니다."
                print(f"\n❌ {fail_msg}")
                self.voice.speak(fail_msg)
        except Exception as e:
            error_msg = f"저장 중 오류가 발생했습니다: {e}"
            print(f"\n❌ {error_msg}")
            self.voice.speak("저장 중 오류가 발생했습니다.")

        time.sleep(1)
    
    def handle_medicine_query(self):
        """약품 질의 처리"""
        # 상자 OCR 안내
        ocr_prompt = "만약 약품 상자가 있다면 카메라에 상자를 보여준 뒤 S 버튼을 눌러주세요. 만약 없다면 그냥 S 버튼을 눌러주세요."
        print(f"\n🔊 {ocr_prompt}")
        self.voice.speak(ocr_prompt)

        # 사용자 입력 대기
        user_input = input("\n📷 준비되면 S를 입력하세요: ").strip().upper()

        if user_input == 'S':
            # 카메라로 사진 촬영
            if self.capture_medicine_image():
                print(f"\n📸 이미지 촬영 완료: {self.med_img}")
                ocr_text, _ = self.ocr.extract_text_with_preprocessing(self.med_img)

                if ocr_text:
                    self.ocr_context = self.ocr.format_for_llm(ocr_text)
                    print("✅ 약품 정보가 추출되었습니다.")
                else:
                    print("⚠️ 이미지에서 텍스트를 추출할 수 없었습니다.")
                    self.ocr_context = None
            else:
                print("❌ 카메라 촬영에 실패했습니다. OCR을 건너뜁니다.")
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
    
    def handle_medicine_view(self):
        """저장된 약품 조회 처리"""
        print("\n📋 저장된 약품을 조회합니다...")
        
        try:
            if not self.search_index:
                error_msg = "저장된 약품이 없습니다."
                print(f"❌ {error_msg}")
                self.voice.speak(error_msg)
                return
            
            # 일반의약품과 처방약으로 분류
            general_medicines = []
            given_medicines = []
            
            for item in self.search_index:
                if item['source'] == 'general':
                    general_medicines.append(item['name'])
                elif item['source'] == 'given':
                    given_medicines.append(item['name'])
            
            # 음성 메시지 생성
            message_parts = []
            
            if general_medicines:
                general_list = ", ".join(general_medicines)
                general_msg = f"일반의약품에 {general_list}이 있습니다."
                message_parts.append(general_msg)
            
            if given_medicines:
                given_list = ", ".join(given_medicines)
                given_msg = f"처방약에 {given_list}이 있습니다."
                message_parts.append(given_msg)
            
            if not message_parts:
                no_medicine_msg = "저장된 약품이 없습니다."
                print(f"❌ {no_medicine_msg}")
                self.voice.speak(no_medicine_msg)
                return
            
            # 화면 출력
            print("\n" + "="*60)
            print("📋 저장된 약품 목록")
            print("-"*60)
            
            if general_medicines:
                print(f"💊 일반의약품 ({len(general_medicines)}개):")
                for medicine in general_medicines:
                    print(f"  - {medicine}")
                print()
            
            if given_medicines:
                print(f"💉 처방약 ({len(given_medicines)}개):")
                for medicine in given_medicines:
                    print(f"  - {medicine}")
                print()
            
            print("="*60)
            
            # 음성 출력
            full_message = " ".join(message_parts)
            print(f"\n🔊 {full_message}")
            self.voice.speak(full_message)
            
        except Exception as e:
            error_msg = f"조회 중 오류가 발생했습니다: {e}"
            print(f"❌ {error_msg}")
            self.voice.speak("조회 중 오류가 발생했습니다.")

    def handle_medicine_retrieve(self):
        """약품 꺼내기 처리 (초고속 검색)"""
        print("\n💊 약품 꺼내기 프로세스를 시작합니다...")
        
        # 음성으로 약품 이름 입력 받기
        retrieve_prompt = "어떤 약을 꺼내시겠습니까? 약품 이름을 말씀해주세요."
        print(f"\n🔊 {retrieve_prompt}")
        self.voice.speak(retrieve_prompt)
        
        # 음성으로 약품명 받기 (10초)
        print("\n🎤 꺼낼 약품명을 음성으로 말씀해주세요...")
        user_input = self.voice.listen(duration=10)
        
        if not user_input:
            no_input_msg = "음성을 인식할 수 없었습니다. 다시 시도해주세요."
            print(f"❌ {no_input_msg}")
            self.voice.speak(no_input_msg)
            return
        
        print(f"\n🎯 인식된 음성: {user_input}")
        
        try:
            if not self.search_index:
                error_msg = "저장된 약품이 없습니다."
                print(f"❌ {error_msg}")
                self.voice.speak(error_msg)
                return
            
            print("🔍 스마트 검색 중...")
            
            # 1단계: 직접적인 약품명 검색 (가장 빠름)
            exact_match, candidates = self.fast_search_medicine(user_input)
            
            if exact_match:
                success_msg = f"{exact_match['row']}칸에 있는 {exact_match['name']}을 열었습니다"
                print(f"\n✅ [직접 매치] {success_msg}")
                self.voice.speak(success_msg)
                return
            
            elif len(candidates) == 1:
                match = candidates[0]
                success_msg = f"{match['row']}칸에 있는 {match['name']}을 열었습니다"
                print(f"\n✅ [키워드 매치] {success_msg}")
                self.voice.speak(success_msg)
                return
            
            # 2단계: 직접 매치 실패시에만 GPT 자연어 처리
            print("🧠 자연어 처리 중...")
            sentence_match, sentence_candidates = self.extract_medicine_from_sentence(user_input)
            
            if sentence_match:
                success_msg = f"{sentence_match['row']}칸에 있는 {sentence_match['name']}을 열었습니다"
                print(f"\n✅ [GPT 분석] {success_msg}")
                self.voice.speak(success_msg)
                return
            
            elif len(sentence_candidates) == 1:
                match = sentence_candidates[0]
                success_msg = f"{match['row']}칸에 있는 {match['name']}을 열었습니다"
                print(f"\n✅ [GPT 분석] {success_msg}")
                self.voice.speak(success_msg)
                return
            
            # 3단계: 여러 후보 중 최종 선택 (마지막 수단)
            final_candidates = candidates if candidates else sentence_candidates
            
            if len(final_candidates) > 1:
                print(f"🎯 {len(final_candidates)}개 후보에서 최종 선택 중...")
                
                candidate_names = [c['name'] for c in final_candidates]
                extract_prompt = f""""{user_input}" → 후보: {', '.join(candidate_names)}
가장 적절한 약품명 하나만 출력. 없으면 "없음"."""
                
                extracted_name = self.llm.query(extract_prompt, None).strip()
                
                if extracted_name != "없음":
                    for candidate in final_candidates:
                        if (extracted_name in candidate['name'] or 
                            candidate['name'] in extracted_name):
                            success_msg = f"{candidate['row']}칸에 있는 {candidate['name']}을 열었습니다"
                            print(f"\n✅ [최종 선택] {success_msg}")
                            self.voice.speak(success_msg)
                            return
            
            # 모든 검색 실패
            not_found_msg = "요청하신 약품을 찾을 수 없습니다."
            print(f"❌ {not_found_msg}")
            self.voice.speak(not_found_msg)
                
        except Exception as e:
            error_msg = f"꺼내기 중 오류가 발생했습니다: {e}"
            print(f"❌ {error_msg}")
            self.voice.speak("꺼내기 중 오류가 발생했습니다.")
    
    def start_voice_interaction(self):
        """음성 상호작용 시작"""
        # 인사말
        greeting = "어떤 것을 원하십니까? 1번 약품 저장, 2번 약품 질의, 3번 약품 꺼내기, 4번 약품 조회"
        print(f"\n🔊 {greeting}")
        self.voice.speak(greeting)
        
        # 사용자 선택 받기 (키보드 입력으로 간소화)
        print("\n선택해주세요:")
        print("  1: 약품 저장")
        print("  2: 약품 질의")
        print("  3: 약품 꺼내기")
        print("  4: 약품 조회")
        
        choice = input("\n번호 입력 (1, 2, 3 또는 4): ").strip()
        
        if choice == '1':
            self.handle_medicine_storage()
        elif choice == '2':
            self.handle_medicine_query()
        elif choice == '3':
            self.handle_medicine_retrieve()
        elif choice == '4':
            self.handle_medicine_view()
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
        "카메라": check_camera_availability()
    }

    all_ok = True
    for item, status in requirements.items():
        if status:
            print(f"  ✅ {item}")
        else:
            print(f"  ❌ {item} - {'설정 필요' if 'API' in item else '사용 불가'}")
            if item != "카메라":
                all_ok = False

    if not all_ok:
        print("\n⚠️ 일부 요구사항이 충족되지 않았습니다.")
        print("다음 사항을 확인해주세요:")

        if not requirements["OPENAI_API_KEY"]:
            print("1. .env 파일에 OPENAI_API_KEY를 설정하세요")

        if not requirements["e_data.json"] or not requirements["n_data.json"]:
            print("2. 의약품 데이터 파일(e_data.json, n_data.json)을 준비하세요")
            print("   (없어도 실행은 가능하지만 RAG 기능이 제한됩니다)")

        print("\n계속 진행하시겠습니까? (y/n)")
        if input().strip().lower() != 'y':
            return False

    if not requirements["카메라"]:
        print("\n⚠️ 카메라를 사용할 수 없지만 시스템은 실행됩니다.")
        print("   (카메라 기능이 제한되지만 다른 기능은 정상 작동합니다)")

    return True

def check_camera_availability():
    """카메라 사용 가능 여부 확인"""
    try:
        # 여러 카메라 인덱스 시도
        for i in range(3):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.release()
                return True
        return False
    except:
        return False

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