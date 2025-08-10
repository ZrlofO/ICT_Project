"""
LLM 모듈 - RAG를 적용한 의약품 정보 질의응답
"""

import json
import os
from typing import Optional
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

load_dotenv()

class LLMModule:
    def __init__(self, e_data_path="e_data.json", n_data_path="n_data.json"):
        """
        LLM RAG 모듈 초기화
        
        Args:
            e_data_path: 의약품 허가정보 JSON 파일 경로
            n_data_path: 의약품 개요정보 JSON 파일 경로
        """
        print("🤖 LLM 모듈을 초기화하는 중...")
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        self.e_data_path = e_data_path
        self.n_data_path = n_data_path
        
        # OpenAI 설정
        os.environ["OPENAI_API_KEY"] = self.api_key
        
        # LLM 설정
        self.llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=2000
        )
        
        # 임베딩 모델 설정
        self.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            embed_batch_size=100
        )
        
        # Settings 구성
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 100
        
        # 인덱스 초기화
        self.index = None
        self.query_engine = None
        
        # 인덱스 구축
        self._build_index()
        
        print("✅ LLM 모듈 준비 완료!")
    
    def _build_index(self):
        """벡터 인덱스 구축"""
        index_path = "./medicine_index"
        
        # 기존 인덱스 로드 시도
        if os.path.exists(index_path):
            print("  📂 기존 인덱스를 로드하는 중...")
            from llama_index.core import load_index_from_storage
            storage_context = StorageContext.from_defaults(persist_dir=index_path)
            self.index = load_index_from_storage(storage_context)
        else:
            print("  🔨 새로운 인덱스를 구축하는 중...")
            documents = self._load_data()
            
            if documents:
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    show_progress=True
                )
                self.index.storage_context.persist(persist_dir=index_path)
            else:
                print("  ⚠️ 의약품 데이터를 찾을 수 없습니다. RAG 없이 동작합니다.")
                return
        
        # 쿼리 엔진 설정
        if self.index:
            self._setup_query_engine()
    
    def _load_data(self):
        """의약품 데이터 로드"""
        documents = []
        
        # e_data.json 로드
        if os.path.exists(self.e_data_path):
            try:
                with open(self.e_data_path, 'r', encoding='utf-8') as f:
                    e_data = json.load(f)
                    for medicine in e_data.get('medicines', []):
                        text = self._format_medicine_data(medicine, "permit")
                        if text:
                            doc = Document(
                                text=text,
                                metadata={
                                    "source": "permit",
                                    "item_name": medicine.get("itemName", "")[:100]
                                }
                            )
                            documents.append(doc)
            except Exception as e:
                print(f"  ⚠️ e_data.json 로드 실패: {e}")
        
        # n_data.json 로드
        if os.path.exists(self.n_data_path):
            try:
                with open(self.n_data_path, 'r', encoding='utf-8') as f:
                    n_data = json.load(f)
                    for medicine in n_data.get('medicines', []):
                        text = self._format_medicine_data(medicine, "overview")
                        if text:
                            doc = Document(
                                text=text,
                                metadata={
                                    "source": "overview",
                                    "item_name": (medicine.get("item_name") or 
                                                medicine.get("ITEM_NAME", ""))[:100]
                                }
                            )
                            documents.append(doc)
            except Exception as e:
                print(f"  ⚠️ n_data.json 로드 실패: {e}")
        
        return documents
    
    def _format_medicine_data(self, medicine, source_type):
        """의약품 데이터를 텍스트로 포맷"""
        text_parts = []
        
        if source_type == "permit":
            # 허가정보 포맷
            fields = [
                ("itemName", "의약품명"),
                ("entpName", "제조사"),
                ("efcyQesitm", "효능효과"),
                ("useMethodQesitm", "용법용량"),
                ("atpnWarnQesitm", "주의사항 경고"),
                ("atpnQesitm", "주의사항"),
                ("intrcQesitm", "상호작용"),
                ("seQesitm", "부작용"),
                ("depositMethodQesitm", "보관방법")
            ]
            
            for field, label in fields:
                if medicine.get(field):
                    text_parts.append(f"{label}: {medicine[field]}")
        
        else:  # overview
            # 개요정보 포맷
            fields = [
                (["item_name", "ITEM_NAME"], "의약품명"),
                (["entp_name", "ENTP_NAME"], "제조사"),
                (["chart", "CHART"], "성상"),
                (["drug_shape", "DRUG_SHAPE"], "의약품 모양"),
                (["color_class1", "COLOR_CLASS1"], "색상 앞"),
                (["color_class2", "COLOR_CLASS2"], "색상 뒤"),
                (["class_name", "CLASS_NAME"], "약품 분류"),
                (["etc_otc_name", "ETC_OTC_NAME"], "전문/일반")
            ]
            
            for field_names, label in fields:
                for field in field_names:
                    if medicine.get(field):
                        text_parts.append(f"{label}: {medicine[field]}")
                        break
        
        return "\n".join(text_parts) if text_parts else None
    
    def _setup_query_engine(self):
        """쿼리 엔진 설정"""
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10,
        )
        
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
        )
        
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.3)
            ],
        )
    
    def query(self, question: str, ocr_context: Optional[str] = None):
        """
        질문에 대한 답변 생성
        
        Args:
            question: 사용자 질문
            ocr_context: OCR로 추출된 약품 정보 (있는 경우)
            
        Returns:
            생성된 답변
        """
        # 질문 향상
        enhanced_question = self._enhance_question(question, ocr_context)
        
        print(f"💭 질문 처리 중: '{question[:50]}...'")
        if ocr_context:
            print("  📄 OCR 컨텍스트 포함")
        
        try:
            if self.query_engine:
                # RAG를 사용한 답변 생성
                response = self.query_engine.query(enhanced_question)
                answer = str(response)
            else:
                # RAG 없이 직접 LLM 사용
                prompt = enhanced_question
                response = self.llm.complete(prompt)
                answer = str(response)
            
            if not answer or not answer.strip():
                answer = "죄송합니다. 관련된 정보를 찾을 수 없습니다."
            
            return answer
            
        except Exception as e:
            print(f"❌ 질문 처리 중 오류: {e}")
            return "죄송합니다. 질문 처리 중 오류가 발생했습니다."
    
    def _enhance_question(self, question: str, ocr_context: Optional[str] = None):
        """질문 향상 -"""
        enhanced = "의약품 정보 데이터베이스를 바탕으로 다음 질문에 답해주세요.\n\n"
        
        # OCR 컨텍스트 추가
        if ocr_context:
            enhanced += f"사용자가 입력으로 넣은 의약품 정보입니다 : {ocr_context}\n\n"
        
        enhanced += f"질문: {question}\n\n"
        enhanced += "답변은 일반인이 이해하기 쉽게 설명해주시고, 중요한 주의사항이 있다면 반드시 포함해주세요. 핵심 요점을 먼저 이야기하고, 가급적 물어본 질문의 핵심 내용만 대답해주세요. 질문에 포함되지 않는 내용은 답변하지 마세요."
        
        # 특정 키워드에 따른 추가 지시
        if "부작용" in question:
            enhanced += " 부작용과 이상반응 정보를 중심으로 설명해주세요."
        elif "용법" in question or "용량" in question:
            enhanced += " 알약의 경우 몇 알 단위로, 액체의 경우 mg 단위로 설명해주세요. 사용법 또는 용량을 중심으로 설명해주세요."
        elif "상호작용" in question or "같이" in question:
            enhanced += " 약물 상호작용과 병용 금기 정보를 중심으로 설명해주세요."
        
        return enhanced