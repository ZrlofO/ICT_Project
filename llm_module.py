"""
LLM 모듈 - RAG를 적용한 의약품 정보 질의응답
"""

import json
import os
import csv
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
                SimilarityPostprocessor(similarity_cutoff=0.5)
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
        print(f"💭 질문 처리 중: '{question[:50]}...'")
        if ocr_context:
            print("  📄 OCR 컨텍스트 포함")
        
        try:
            if self.query_engine and self.index:
                # 1. 먼저 벡터 검색으로 관련 문서 가져오기
                print("🔍 벡터 검색 수행 중...")
                retriever = VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=10,  # 더 많이 가져와서 필터링
                )
                retrieved_nodes = retriever.retrieve(question)
                
                # 2. 검색된 문서들의 유사도 점수 추출 및 필터링
                filtered_docs = []
                similarity_threshold = 0.5  # 0.5보다 낮게 설정하여 더 많은 관련 문서 포함
                
                for i, node in enumerate(retrieved_nodes):
                    try:
                        # 유사도 점수 추출 (여러 방법 시도)
                        score = None
                        if hasattr(node, 'score') and node.score is not None:
                            score = float(node.score)
                        elif hasattr(node, 'metadata') and node.metadata and 'score' in node.metadata:
                            score = float(node.metadata['score'])
                        else:
                            # 검색 순서 기반으로 점수 추정 (첫 번째가 가장 관련성 높음)
                            score = max(0.1, 1.0 - (i * 0.1))  # 1.0, 0.9, 0.8, ..., 최소 0.1
                        
                        # 점수 검증
                        if score is None or score < 0:
                            score = max(0.1, 1.0 - (i * 0.15))
                        
                        # 텍스트 추출 및 검증
                        doc_text = ""
                        if hasattr(node, 'text') and node.text:
                            doc_text = str(node.text).strip()[:800]
                        elif hasattr(node, 'content') and node.content:
                            doc_text = str(node.content).strip()[:800]
                        
                        if not doc_text or len(doc_text.strip()) < 10:
                            print(f"  ⚠️ 빈 문서 건너뜀 (인덱스: {i})")
                            continue
                        
                        # 메타데이터 안전하게 추출
                        source = 'unknown'
                        if hasattr(node, 'metadata') and node.metadata:
                            source = node.metadata.get('source', 'unknown')
                        
                        # 유사도 임계값 이상인 문서만 선택
                        if score >= similarity_threshold:
                            filtered_docs.append({
                                'text': doc_text,
                                'score': score,
                                'source': source
                            })
                            print(f"  ✅ 선택된 문서 (유사도: {score:.3f}, 출처: {source}): {doc_text[:80]}...")
                        else:
                            print(f"  ❌ 제외된 문서 (유사도: {score:.3f}): {doc_text[:50]}...")
                            
                    except Exception as node_error:
                        print(f"  ⚠️ 노드 처리 오류 (인덱스: {i}): {node_error}")
                        continue
                
                # 3. 검색 결과 품질 확인 및 보완
                if not filtered_docs and len(retrieved_nodes) > 0:
                    print("⚠️ 유사도 기준을 만족하는 문서가 없습니다. 기준을 낮춰서 재시도...")
                    # 기준을 낮춰서 재시도
                    for i, node in enumerate(retrieved_nodes[:3]):  # 상위 3개만
                        try:
                            score = 1.0 - (i * 0.2) if not (hasattr(node, 'score') and node.score) else float(node.score)
                            
                            # 텍스트 안전하게 추출
                            doc_text = ""
                            if hasattr(node, 'text') and node.text:
                                doc_text = str(node.text).strip()[:800]
                            elif hasattr(node, 'content') and node.content:
                                doc_text = str(node.content).strip()[:800]
                                
                            if doc_text and len(doc_text.strip()) >= 10:
                                source = 'unknown'
                                if hasattr(node, 'metadata') and node.metadata:
                                    source = node.metadata.get('source', 'unknown')
                                    
                                filtered_docs.append({
                                    'text': doc_text,
                                    'score': score,
                                    'source': source
                                })
                                print(f"  📄 추가된 문서 (유사도: {score:.3f}): {doc_text[:80]}...")
                        except Exception as fallback_error:
                            print(f"  ⚠️ 보완 처리 오류 (인덱스: {i}): {fallback_error}")
                            continue
                
                # 4. 사용자의 복용중인 약품 정보 가져오기
                user_medicines = self._get_user_medicines()
                
                # 5. RAG context 구성 (점수 순으로 정렬)
                filtered_docs.sort(key=lambda x: x['score'], reverse=True)
                rag_docs_text = []
                for doc in filtered_docs:
                    rag_docs_text.append(f"[유사도: {doc['score']:.3f} | 출처: {doc['source']}]\n{doc['text']}")
                
                rag_context = "\n\n---\n\n".join(rag_docs_text) if rag_docs_text else "관련 문서를 찾을 수 없습니다."
                
                print(f"📊 최종 선택된 문서 수: {len(filtered_docs)}개")
                
                # 6. 검색 결과 + OCR context + 사용자 복용약품을 포함한 프롬프트 생성
                enhanced_question = self._enhance_question(question, ocr_context, rag_context, user_medicines)
                
                # 7. LLM으로 직접 답변 생성 (검색 결과 포함)
                print("🤖 RAG 기반 답변 생성 중...")
                
                # 프롬프트 길이 확인 (너무 길면 토큰 제한 초과 가능)
                if len(enhanced_question) > 15000:
                    print(f"⚠️ 프롬프트가 길어서 축약 중... (현재: {len(enhanced_question)}자)")
                    # RAG context를 줄임
                    if len(filtered_docs) > 3:
                        filtered_docs = filtered_docs[:3]
                        rag_docs_text = []
                        for doc in filtered_docs:
                            rag_docs_text.append(f"[유사도: {doc['score']:.3f} | 출처: {doc['source']}]\n{doc['text']}")
                        rag_context = "\n\n---\n\n".join(rag_docs_text)
                        enhanced_question = self._enhance_question(question, ocr_context, rag_context, user_medicines)
                        print(f"📏 프롬프트 축약 완료: {len(enhanced_question)}자")
                
                response = self.llm.complete(enhanced_question)
                answer = str(response).strip()
                
                # 답변 품질 검증
                if not answer or len(answer) < 10:
                    answer = "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다."
                elif "검색된 정보가 제한적" in answer or "관련 정보를 찾을 수 없" in answer:
                    print("ℹ️ LLM이 정보 부족을 인지했습니다.")
                else:
                    print("✅ RAG 기반 답변 생성 완료")
            else:
                # RAG 없이 직접 LLM 사용
                print("!RAG 안사용!")
                user_medicines = self._get_user_medicines()
                enhanced_question = self._enhance_question(question, ocr_context, None, user_medicines)
                response = self.llm.complete(enhanced_question)
                answer = str(response)
            
            if not answer or not answer.strip():
                answer = "죄송합니다. 관련된 정보를 찾을 수 없습니다."
            
            return answer
            
        except Exception as e:
            print(f"❌ 질문 처리 중 오류: {e}")
            return "죄송합니다. 질문 처리 중 오류가 발생했습니다."
    
    def _get_user_medicines(self):
        """사용자가 복용중인 약품 정보 가져오기 (given.csv에서)"""
        user_medicines = []
        
        try:
            if os.path.exists("given.csv"):
                with open("given.csv", 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        if row.get('제목') and row.get('OCR정보'):
                            # OCR 정보를 더 적절히 정리
                            title = row['제목'].strip()
                            ocr_info = row['OCR정보'].strip()
                            
                            # OCR 정보가 너무 길면 중요한 부분만 추출
                            if len(ocr_info) > 300:
                                # 주요 키워드가 포함된 부분만 추출
                                keywords = ['용법', '용량', '성분', '주의', '부작용', '효능', '효과']
                                important_parts = []
                                lines = ocr_info.split('\n')
                                
                                for line in lines:
                                    if any(keyword in line for keyword in keywords) or len(line.strip()) < 50:
                                        important_parts.append(line.strip())
                                        if len('\n'.join(important_parts)) > 250:
                                            break
                                
                                ocr_summary = '\n'.join(important_parts[:5])  # 최대 5줄
                                if len(ocr_summary) > 250:
                                    ocr_summary = ocr_summary[:250] + "..."
                            else:
                                ocr_summary = ocr_info
                            
                            medicine_info = f"- {title}: {ocr_summary}"
                            user_medicines.append(medicine_info)
                            
                if user_medicines:
                    print(f"💊 사용자 복용중인 약품 {len(user_medicines)}개 발견")
                else:
                    print("💊 저장된 처방약이 없습니다.")
                    
        except Exception as e:
            print(f"⚠️ 사용자 약품 정보 로드 실패: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            
        return user_medicines
    
    def _enhance_question(self, question: str, ocr_context: Optional[str] = None, rag_context: Optional[str] = None, user_medicines: Optional[list] = None):
        """
        의약품 질문 프롬프트 생성 (RAG + OCR + 사용자 복용약품 활용)
        """
        # System 역할: 모델의 기본 행동 정의
        system_instruction = (
            "당신은 의약품 정보 데이터베이스를 기반으로 질문에 답하는 의료 도우미입니다.\n"
            "답변은 반드시 일반인이 이해하기 쉽게 작성하세요.\n"
            "핵심 요점을 먼저 말하고, 중요한 주의사항은 반드시 포함하세요.\n"
            "질문에 포함되지 않은 불필요한 내용은 답변하지 마세요.\n"
            "사용자가 현재 복용중인 약품이 있다면 상호작용이나 주의사항을 반드시 고려하세요.\n"
        )

        # Context: RAG 검색 결과 + OCR 정보 + 사용자 복용약품
        context_block = "### 참고 정보\n"
        
        # RAG 검색 결과 추가
        if rag_context:
            context_block += f"\n**데이터베이스 검색 결과:**\n{rag_context}\n"
        
        # OCR 정보 추가
        if ocr_context:
            context_block += f"\n**사용자가 입력한 의약품 OCR 정보:**\n{ocr_context}\n"
        
        # 사용자 복용중인 약품 정보 추가
        if user_medicines and len(user_medicines) > 0:
            context_block += f"\n**사용자가 현재 복용중인 약품들:**\n" + "\n".join(user_medicines) + "\n"

        # 질문
        question_block = f"### 질문\n{question}\n"

        # 출력 형식 지시
        output_instruction = (
            "### 답변 지침\n"
            "1. 위의 검색 결과, OCR 정보, 사용자 복용약품 정보를 모두 종합하여 답변하세요.\n"
            "2. 먼저 핵심 요약을 한 문단으로 제시하세요.\n"
            "3. 이어서 세부 설명을 간단명료하게 덧붙이세요.\n"
            "4. 주의사항이 있다면 반드시 별도 항목으로 강조하세요.\n"
            "5. **중요**: 사용자가 복용중인 약품과의 상호작용, 병용 금기사항, 부작용이 있다면 반드시 별도로 강조하세요.\n"
            "6. 의학적 조언이나 진단은 피하고, 일반적인 의약품 정보만 제공하세요.\n",
            "7. #이나 *같은 특수문자는 사용하지 말고 제공하세요.\n"
        )

        # 특정 키워드 맞춤 지시
        if "부작용" in question:
            output_instruction += "8. 답변은 부작용 및 이상반응 정보를 중심으로 하세요.\n"
        elif "용법" in question or "용량" in question:
            output_instruction += "8. 용법과 용량을 구체적으로 설명하되, 의사의 처방이 우선임을 명시하세요.\n"
        elif "상호작용" in question or "같이" in question or "함께" in question:
            output_instruction += "8. 약물 상호작용과 병용 금기 정보를 중심으로 설명하세요.\n"
        
        # 사용자 복용약품이 있으면 상호작용 검토 지시 추가
        if user_medicines and len(user_medicines) > 0:
            output_instruction += "\n**🚨 특별 주의사항**: \n"
            output_instruction += "위에 나열된 사용자의 현재 복용약품들과 질문하신 약품/내용 간의:\n"
            output_instruction += "- 상호작용 가능성\n"
            output_instruction += "- 병용 금기사항\n" 
            output_instruction += "- 주의사항 및 부작용 증가 위험\n"
            output_instruction += "을 반드시 검토하여 별도 섹션으로 강조해서 답변하세요.\n"

        # 최종 프롬프트 조립
        enhanced_prompt = f"{system_instruction}\n{context_block}\n{question_block}\n{output_instruction}"
        return enhanced_prompt
