"""
약품 저장 모듈 - 약품 분류 및 저장 기능 제공
"""

import os
import csv
import time
from datetime import datetime
from .voice_module import VoiceModule
from .ocr_module import OCRModule
from .llm_module import LLMModule

class StoreModule:
    def __init__(self, voice_module=None, ocr_module=None, llm_module=None):
        """약품 저장 모듈 초기화"""
        print("💾 약품 저장 모듈을 초기화하는 중...")
        
        # 이미 초기화된 모듈들을 받거나 새로 생성
        if voice_module:
            self.voice = voice_module
        else:
            self.voice = VoiceModule()
            
        if ocr_module:
            self.ocr = ocr_module
        else:
            self.ocr = OCRModule()
            
        if llm_module:
            self.llm = llm_module
        else:
            self.llm = LLMModule()
        
        # CSV 파일 경로
        self.general_csv = "user_med_data/general.csv"
        self.given_csv = "user_med_data/given.csv"

        # user_med_data 폴더 생성
        os.makedirs("user_med_data", exist_ok=True)
        
        # CSV 파일 초기화
        self._initialize_csv_files()
        
        print("✅ 약품 저장 모듈 준비 완료!")
    
    def _initialize_csv_files(self):
        """CSV 파일들 초기화 (헤더 생성)"""
        # general.csv 헤더 (일반의약품)
        if not os.path.exists(self.general_csv):
            with open(self.general_csv, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['약품명', '저장날짜'])
        
        # given.csv 헤더 (처방약)
        if not os.path.exists(self.given_csv):
            with open(self.given_csv, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['제목', 'OCR정보', '저장날짜'])
    
    def start_storage_process(self, med_img="scan/capture.jpg"):
        """약품 저장 프로세스 시작 (이미 촬영된 이미지 사용)"""
        return self._process_medicine_image(med_img)
    
    def _process_medicine_image(self, med_img="scan/capture.jpg"):
        """약품 이미지 처리 및 분류"""
        # OCR 처리
        if not os.path.exists(med_img):
            error_msg = "이미지 파일을 찾을 수 없습니다."
            print(f"❌ {error_msg}")
            self.voice.speak(error_msg)
            return False

        print(f"\n📸 이미지 분석 중: {med_img}")
        ocr_text, _ = self.ocr.extract_text_with_preprocessing(med_img)

        if not ocr_text or not ocr_text.strip():
            error_msg = "인식된 것이 없어 저장할 수 없습니다"
            print(f"❌ {error_msg}")
            self.voice.speak(error_msg)
            return False

        print(f"✅ OCR 완료: {ocr_text}")

        # GPT를 통한 약품 분류
        classification = self._classify_medicine(ocr_text)

        if classification['type'] == '기타':
            retry_msg = "약 상자 또는 약봉투를 뒤집어 다시 진행해주세요."
            print(f"⚠️ {retry_msg}")
            self.voice.speak(retry_msg)
            return False  # 더 이상 재시도하지 않음

        # 분류에 따른 처리
        if classification['type'] == '일반의약품':
            return self._handle_general_medicine(classification)
        elif classification['type'] == '처방약':
            return self._handle_prescription_medicine(ocr_text, med_img)

        return False
    
    def _classify_medicine(self, ocr_text):
        """GPT를 통해 약품 분류"""
        print("\n🤔 약품 분류 중...")
        
        classification_prompt = f"""
다음 OCR로 추출된 텍스트를 분석하여 약품을 분류해주세요.

OCR 텍스트:
{ocr_text}

분류 기준:
1. 일반의약품: 일반의약품명이 확실하게 확인되는 경우 (예: 타이레놀, 애드빌, 판콜 등)
2. 처방약: 일반의약품이 드러나있지 않고 약의 성분이나 투약량이나 복약 안내가 드러나있는 경우
3. 기타: 위 두 경우 모두 아닌 경우

응답 형식:
분류: [일반의약품/처방약/기타]
약품명: [일반의약품인 경우만 구체적인 제품명 작성, 나머지는 빈칸]

예시:
- 만약 "타이레놀 500mg" 같은 텍스트가 있다면
  분류: 일반의약품
  약품명: 타이레놀 500mg

- 만약 "아세트아미노펜 복용법..." 같은 텍스트가 있다면
  분류: 처방약

- 만약 약과 관련없는 내용이라면
  분류: 기타
"""
        
        try:
            response = self.llm.query(classification_prompt)
            print(f"📋 GPT 분류 결과: {response}")
            
            # 응답 파싱
            lines = response.split('\n')
            classification_type = None
            medicine_name = None
            
            for line in lines:
                if '분류:' in line:
                    classification_type = line.split('분류:')[1].strip()
                elif '약품명:' in line:
                    medicine_name = line.split('약품명:')[1].strip()
            
            if not classification_type:
                return {'type': '기타', 'name': None}
            
            return {
                'type': classification_type,
                'name': medicine_name if medicine_name else None
            }
            
        except Exception as e:
            print(f"❌ 분류 오류: {e}")
            return {'type': '기타', 'name': None}
    
    def _handle_general_medicine(self, classification):
        """일반의약품 처리"""
        medicine_name = classification.get('name', '알 수 없는 약품')
        
        # 1. 인식 완료 음성 안내
        success_msg = f"{medicine_name} 인식 완료했습니다."
        print(f"\n✅ {success_msg}")
        self.voice.speak(success_msg)
        
        # 2. 중복 체크
        duplicate_info = self._check_duplicate_general_medicine(medicine_name)
        if duplicate_info:
            duplicate_msg = f"{duplicate_info['month']}월 {duplicate_info['day']}일에 {medicine_name}을 이미 저장하셨습니다."
            print(f"\n⚠️ {duplicate_msg}")
            self.voice.speak(duplicate_msg)
            return True  # 중복이지만 정상 종료로 처리
        
        # 3. general.csv에 저장
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            with open(self.general_csv, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([medicine_name, current_date])
            
            print(f"💾 일반의약품 저장 완료: {medicine_name}")
            return True
            
        except Exception as e:
            print(f"❌ 저장 오류: {e}")
            error_msg = "저장 중 오류가 발생했습니다."
            self.voice.speak(error_msg)
            return False
    
    def _handle_prescription_medicine(self, ocr_text, med_img="scan/capture.jpg"):
        """처방약 처리"""
        while True:
            # 1. 이름 입력 요청
            name_request = "어떤 이름으로 저장하시겠습니까?"
            print(f"\n🔊 {name_request}")
            self.voice.speak(name_request)
            
            # 음성으로 답변 받기
            print("\n🎤 음성으로 답변해주세요...")
            user_response = self.voice.listen(duration=10)
            
            if not user_response:
                retry_msg = "답변을 인식할 수 없었습니다. 다시 시도해주세요."
                print(f"❌ {retry_msg}")
                self.voice.speak(retry_msg)
                continue
            
            print(f"📝 인식된 답변: {user_response}")
            
            # 2. GPT를 통해 제목 키워드 추출
            title_keyword = self._extract_title_keyword(user_response)
            
            # 2.5. 중복 체크
            duplicate_info = self._check_duplicate_prescription_medicine(title_keyword)
            if duplicate_info:
                duplicate_msg = f"{duplicate_info['month']}월 {duplicate_info['day']}일에 {title_keyword}을 이미 저장하셨습니다."
                print(f"\n⚠️ {duplicate_msg}")
                self.voice.speak(duplicate_msg)
                return True  # 중복이지만 정상 종료로 처리
            
            # 2.6. 저장 확인 안내
            confirm_msg = f"{title_keyword}로 저장하겠습니다"
            print(f"\n💾 {confirm_msg}")
            self.voice.speak(confirm_msg)
            
            # 3. given.csv에 저장
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            try:
                with open(self.given_csv, 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([title_keyword, ocr_text, current_date])
                
                print(f"✅ 처방약 저장 완료: {title_keyword}")
                
                # 4. 추가 저장 여부 확인
                if self._ask_for_additional_storage():
                    # 5초 안에 'S'를 누르지 않으면 메뉴로 돌아감
                    return self._process_medicine_image(med_img)  # 처방약 1번부터 다시 시작
                else:
                    return True
                    
            except Exception as e:
                print(f"❌ 저장 오류: {e}")
                error_msg = "저장 중 오류가 발생했습니다."
                self.voice.speak(error_msg)
                return False
    
    def _extract_title_keyword(self, user_response):
        """사용자 답변에서 제목 키워드 추출"""
        print("\n🔄 제목 키워드 추출 중...")
        
        keyword_prompt = f"""
사용자의 답변에서 약품을 구분할 수 있는 하나의 간단한 제목 키워드를 추출해주세요.

사용자 답변: "{user_response}"

요구사항:
- 한글 단어 또는 짧은 구문으로 작성
- 약품을 식별할 수 있는 핵심 단어
- 10자 이내로 간단하게
- 특수문자나 불필요한 단어 제거

예시:
입력: "혈압약이요" → 출력: "혈압약"
입력: "감기 때문에 받은 약" → 출력: "감기약"
입력: "당뇨병 치료제입니다" → 출력: "당뇨약"

제목 키워드만 출력해주세요:
"""
        
        try:
            response = self.llm.query(keyword_prompt)
            # 응답에서 키워드만 추출 (줄바꿈이나 여분의 텍스트 제거)
            keyword = response.strip().split('\n')[0].strip()
            
            # 안전장치: 너무 길거나 이상한 경우 기본값 사용
            if len(keyword) > 20 or not keyword:
                keyword = "처방약"
            
            print(f"📌 추출된 키워드: {keyword}")
            return keyword
            
        except Exception as e:
            print(f"❌ 키워드 추출 오류: {e}")
            return "처방약"  # 기본값
    
    def _check_duplicate_general_medicine(self, medicine_name):
        """일반의약품 중복 체크"""
        if not os.path.exists(self.general_csv):
            return None
            
        try:
            with open(self.general_csv, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row.get('약품명') and row['약품명'].strip() == medicine_name.strip():
                        # 날짜 파싱 (YYYY-MM-DD HH:MM:SS 형식)
                        date_str = row.get('저장날짜', '')
                        try:
                            date_obj = datetime.strptime(date_str.split()[0], "%Y-%m-%d")
                            return {
                                'month': date_obj.month,
                                'day': date_obj.day,
                                'date': date_str
                            }
                        except:
                            return {
                                'month': '?',
                                'day': '?',
                                'date': date_str
                            }
            return None
        except Exception as e:
            print(f"⚠️ 중복 체크 오류: {e}")
            return None
    
    def _check_duplicate_prescription_medicine(self, title_keyword):
        """처방약 중복 체크"""
        if not os.path.exists(self.given_csv):
            return None
            
        try:
            with open(self.given_csv, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row.get('제목') and row['제목'].strip() == title_keyword.strip():
                        # 날짜 파싱 (YYYY-MM-DD HH:MM:SS 형식)
                        date_str = row.get('저장날짜', '')
                        try:
                            date_obj = datetime.strptime(date_str.split()[0], "%Y-%m-%d")
                            return {
                                'month': date_obj.month,
                                'day': date_obj.day,
                                'date': date_str
                            }
                        except:
                            return {
                                'month': '?',
                                'day': '?',
                                'date': date_str
                            }
            return None
        except Exception as e:
            print(f"⚠️ 중복 체크 오류: {e}")
            return None
    
    def _ask_for_additional_storage(self):
        """추가 저장 여부 확인"""
        additional_msg = "같은 처방약에 대해서 추가적으로 저장할 것이 있나요?"
        print(f"\n❓ {additional_msg}")
        self.voice.speak(additional_msg)
        
        print("\n⏰ 5초 안에 'S'를 누르면 추가 저장, 아니면 메뉴로 돌아갑니다...")
        
        # Windows용 5초 타이머 구현
        try:
            import msvcrt
            
            start_time = time.time()
            while time.time() - start_time < 5:
                remaining = 5 - int(time.time() - start_time)
                print(f"\r⏱️ 남은 시간: {remaining}초 (S키를 누르세요)  ", end="", flush=True)
                
                # Windows에서 키 입력 확인
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').upper()
                    if key == 'S':
                        print(f"\n✅ 추가 저장을 시작합니다.")
                        return True
                
                time.sleep(0.1)
            
            print(f"\n⏰ 시간 초과. 메뉴로 돌아갑니다.")
            return False
            
        except ImportError:
            # Windows가 아닌 경우 또는 msvcrt를 사용할 수 없는 경우
            print("\n⏰ 5초 후 자동으로 메뉴로 돌아갑니다...")
            time.sleep(5)
            return False
    
    def get_stored_medicines(self):
        """저장된 약품 목록 조회"""
        result = {
            'general': [],
            'prescription': []
        }
        
        # 일반의약품 조회
        if os.path.exists(self.general_csv):
            try:
                with open(self.general_csv, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        result['general'].append(row)
            except Exception as e:
                print(f"❌ 일반의약품 조회 오류: {e}")
        
        # 처방약 조회
        if os.path.exists(self.given_csv):
            try:
                with open(self.given_csv, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        result['prescription'].append(row)
            except Exception as e:
                print(f"❌ 처방약 조회 오류: {e}")
        
        return result

# 테스트 함수
def test_store_module():
    """저장 모듈 테스트"""
    print("저장 모듈 테스트 시작...")
    
    try:
        store = StoreModule()
        
        # 저장 프로세스 테스트
        result = store.start_storage_process()
        
        if result:
            print("✅ 저장 프로세스 완료!")
            
            # 저장된 데이터 확인
            medicines = store.get_stored_medicines()
            print(f"\n저장된 일반의약품: {len(medicines['general'])}개")
            print(f"저장된 처방약: {len(medicines['prescription'])}개")
        else:
            print("❌ 저장 프로세스 실패")
            
    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_store_module()