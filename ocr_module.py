"""
OCR 모듈 - PaddleOCR을 사용한 텍스트 추출
"""

import cv2
from paddleocr import PaddleOCR
import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class OCRModule:
    def __init__(self):
        """OCR 모듈 초기화"""
        print("OCR 모듈을 초기화하는 중...")
        try:
            # use_textline_orientation=True: 회전된 텍스트 처리 (새 API)
            # lang='korean': 한국어 지원
            self.ocr_model = PaddleOCR(use_textline_orientation=True, lang='korean')
            print("OCR 모듈 준비 완료!")
        except Exception as e:
            print(f"OCR 모듈 초기화 실패: {e}")
            raise
    
    def extract_text_from_image(self, img_path):
        """
        이미지에서 텍스트 추출
        
        Args:
            img_path: 이미지 파일 경로
            
        Returns:
            tuple: (추출된 전체 텍스트, 상세 결과 리스트)
        """
        if not os.path.exists(img_path):
            print(f"이미지 파일을 찾을 수 없습니다: {img_path}")
            return None, None
        
        print(f"이미지에서 텍스트를 추출하는 중: {img_path}")
        
        try:
            # OCR 실행 (새 API 사용)
            result = self.ocr_model.predict(img_path)
            
            # PaddleOCR 결과 구조 확인
            print(f"PaddleOCR 결과 타입: {type(result)}")
            print(f"PaddleOCR 전체 결과: {result}")
            
            # 새 API는 딕셔너리 형태로 결과를 반환
            if isinstance(result, list) and len(result) > 0:
                # 첫 번째 요소가 딕셔너리라면
                if isinstance(result[0], dict):
                    result_dict = result[0]
                    
                    # 실제 텍스트는 'rec_texts' 키에 있음
                    if 'rec_texts' in result_dict:
                        rec_texts = result_dict['rec_texts']
                        rec_scores = result_dict.get('rec_scores', [])
                        rec_polys = result_dict.get('rec_polys', [])
                        
                        print(f"추출된 텍스트: {rec_texts}")
                        print(f"신뢰도 점수: {rec_scores}")
                        
                        if rec_texts and len(rec_texts) > 0:
                            extracted_texts = []
                            detailed_results = []
                            
                            for i, text in enumerate(rec_texts):
                                confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                                bbox = rec_polys[i] if i < len(rec_polys) else []
                                
                                if text and isinstance(text, str):
                                    text = text.strip()
                                    if text and confidence > 0.3:  # 신뢰도 30% 이상만 사용
                                        extracted_texts.append(text)
                                        detailed_results.append({
                                            'text': text,
                                            'confidence': confidence,
                                            'bbox': bbox
                                        })
                                        print(f"  [{i+1}] '{text}' (신뢰도: {confidence:.2%})")
                                    else:
                                        if not text:
                                            print(f"  [{i+1}] 빈 텍스트 건너뜀")
                                        else:
                                            print(f"  [{i+1}] 낮은 신뢰도로 건너뜀: '{text}' ({confidence:.2%})")
                            
                            # 전체 텍스트 결합
                            if extracted_texts:
                                combined_text = " ".join(extracted_texts)
                                print(f"추출된 전체 텍스트: {combined_text}")
                                return combined_text, detailed_results
                            else:
                                print("신뢰할 수 있는 텍스트를 찾을 수 없습니다.")
                                return None, None
                    else:
                        print("결과에 'rec_texts' 키가 없습니다.")
                        return None, None
                else:
                    print("결과 형태가 예상과 다릅니다.")
                    return None, None
            else:
                print("OCR 결과가 비어있습니다.")
                return None, None
                
        except Exception as e:
            print(f"OCR 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def preprocess_image(self, img_path):
        """
        이미지 전처리 (OCR 성능 향상을 위해)
        
        Args:
            img_path: 이미지 파일 경로
            
        Returns:
            전처리된 이미지 또는 None
        """
        try:
            # 이미지 로드
            img = cv2.imread(img_path)
            if img is None:
                print(f"이미지를 로드할 수 없습니다: {img_path}")
                return None
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 노이즈 제거
            denoised = cv2.medianBlur(gray, 3)
            
            # 대비 향상
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # 이진화 (적응적 임계값)
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            return binary
            
        except Exception as e:
            print(f"이미지 전처리 실패: {e}")
            return None
    
    def extract_text_with_preprocessing(self, img_path):
        """
        전처리된 이미지에서 텍스트 추출
        
        Args:
            img_path: 이미지 파일 경로
            
        Returns:
            tuple: (추출된 전체 텍스트, 상세 결과 리스트)
        """
        # 원본 이미지로 먼저 시도
        text, details = self.extract_text_from_image(img_path)
        
        if text and len(text.strip()) > 0:
            return text, details
        
        print("원본 이미지에서 텍스트 추출 실패, 전처리 후 재시도...")
        
        # 전처리된 이미지로 재시도
        preprocessed_img = self.preprocess_image(img_path)
        if preprocessed_img is not None:
            try:
                # 임시 파일로 저장
                temp_path = "temp_preprocessed.jpg"
                cv2.imwrite(temp_path, preprocessed_img)
                
                # OCR 재시도
                text, details = self.extract_text_from_image(temp_path)
                
                # 임시 파일 삭제
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                return text, details
                
            except Exception as e:
                print(f"전처리된 이미지 OCR 실패: {e}")
        
        return None, None
    
    def format_for_llm(self, ocr_text):
        """
        OCR 결과를 LLM에 전달하기 위한 포맷으로 변환
        
        Args:
            ocr_text: OCR로 추출된 텍스트
            
        Returns:
            포맷된 텍스트
        """
        if not ocr_text:
            return None
        
        # 텍스트 정리
        cleaned_text = ocr_text.strip()
        if not cleaned_text:
            return None
        
        formatted = f"""[약품 상자에서 추출된 정보]
{cleaned_text}

위 정보는 사용자가 제시한 약품 상자에서 OCR로 추출한 내용입니다.
이 정보를 참고하여 질문에 답변해주세요."""
        
        return formatted

# 테스트 함수
def test_ocr():
    """OCR 모듈 테스트"""
    print("OCR 모듈 테스트 시작...")
    
    try:
        ocr = OCRModule()
        
        # 테스트 이미지 경로
        test_image = "test/aronamin.jpg"
        
        if os.path.exists(test_image):
            print(f"\n테스트 이미지: {test_image}")
            text, details = ocr.extract_text_with_preprocessing(test_image)
            
            if text:
                print(f"\n추출 성공!")
                print(f"텍스트: {text}")
                print(f"세부사항: {len(details)}개 텍스트 영역")
                
                # LLM 포맷 테스트
                formatted = ocr.format_for_llm(text)
                if formatted:
                    print(f"\nLLM 포맷:")
                    print(formatted)
                    
            else:
                print(f"\n텍스트 추출 실패")
        else:
            print(f"\n테스트 이미지를 찾을 수 없습니다: {test_image}")
            
    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ocr()