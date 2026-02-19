from typing import Dict, Optional, List
import json
from src.retriever import SectionRetriever
from src.llm_service import GeminiLLMService
from typing import Dict, Optional, List
import json
import numpy as np
from src.retriever import SectionRetriever
from src.llm_service import GeminiLLMService
from src.schema import DiagnosisResult, FailureMode, Citation

class StructuralAgent:
    def __init__(self, retriever: SectionRetriever, llm_service: GeminiLLMService):
        self.retriever = retriever
    def __init__(self, retriever: SectionRetriever, llm_service: GeminiLLMService):
        self.retriever = retriever
        self.llm_service = llm_service
        
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
            print("CrossEncoder initialized successfully.")
        except Exception as e:
            print(f"Warning: Could not initialize CrossEncoder: {e}")
            self.cross_encoder = None

    def _format_query(self, input_data: Dict[str, str]) -> str:
        """
        Converts structured input into a natural language query for retrieval.
        """
        query_parts = []
        if "material" in input_data:
            query_parts.append(f"{input_data['material']}")
        if "orientation" in input_data:
            query_parts.append(f"{input_data['orientation']} cracks")
        if "location" in input_data:
            query_parts.append(f"in {input_data['location']}")
        if "description" in input_data:
            query_parts.append(f"{input_data['description']}")
            
        return " ".join(query_parts)

    def _calculate_severity(self, width_str: Optional[str]) -> tuple[str, str]:
        """
        Parses crack width and returns (Severity, Scope).
        Rules:
        < 1.0mm: "Fine" -> Scope: Insignificant/Slight
        1.0mm - 5.0mm (inclusive): "Moderate" -> Scope: Moderate
        > 5.0mm: "Severe" -> Scope: Heavy/Extreme
        """
        if not width_str:
            return "Unknown", "All Levels"
            
        try:
            # Simple parsing for "5mm", "0.2 inches", etc.
            # Normalize to mm
            import re
            match = re.search(r"([\d\.]+)\s*(mm|in|inch)", width_str.lower())
            if not match:
                return "Unknown", "All Levels"
                
            val = float(match.group(1))
            unit = match.group(2)
            
            width_mm = val
            if "in" in unit: # inches conversion
                width_mm = val * 25.4
                
            if width_mm < 1.0:
                return "Fine", "Insignificant or Slight"
            elif width_mm <= 5.0:
                return "Moderate", "Moderate"
            else:
                return "Severe", "Heavy or Extreme"
                
        except Exception as e:
            print(f"Error parsing width: {e}")
            return "Unknown", "All Levels"

    def diagnose(self, input_data: Dict[str, str]) -> Optional[DiagnosisResult]:
        """
        Diagnoses the failure mode by comparing observations against multiple retrieved Classification Guides.
        """
        # 1. Retrieve relevant sections
        query = self._format_query(input_data)
        print(f"Retrieving documents for query: '{query}'")
        
        # Retrieve more results to ensure we catch the right guide even if it's ranked 3rd or 4th
        # Note: Not using is_failure_mode filter since documents may not have this metadata field
        results = self.retriever.query_with_score(
            query, 
            n_results=8, 
            filters=None  # Removed filter - search all documents
        )
        
        if not results:
            print("No relevant documents found.")
            return None

        # 2. Identify Candidates (Guides) vs Context
        candidates = []
        titles = []
        
        for doc, score in results:
            title = doc.metadata.get("title", "").lower()
            titles.append(title)
            candidates.append(doc)
        
        # print("\nCandidates:")
        # for i, title in enumerate(titles):
        #     print(f"{i+1}. {title}")    

        # 3. Construct Prompt with Multiple Candidates
        candidates_text = ""
        for i, doc in enumerate(candidates):
            candidates_text += f"-- CANDIDATE {i+1} --\n"
            candidates_text += f"Title: {doc.metadata.get('title')}\n"
            candidates_text += f"ID: {doc.metadata.get('id')}\n"
            candidates_text += f"Content: {doc.page_content}\n\n"


        system_prompt = (
            "You are a structural engineering expert on FEMA 306. "
            "Your task is to identify the correct 'Failure Mode' from the provided Candidates based on the Field Observations. "
            "You must rely strictly on the 'Classification Guides' (tables/descriptions) in the text. "
            "CRITICAL: \n"
            "- 'X-cracking' or 'Diagonal' step-cracks typically indicate 'Diagonal Tension' (often Section 7.2.10).\n"
            "- 'Horizontal' cracks typically indicate 'Rocking' or 'Bed Joint Sliding'.\n"
            "- 'Vertical' cracks typically indicate 'Spandrel' or 'Flexural' issues.\n"
            "Ignore candidates where the 'Orientation' listed in the text does not match the observations."
        )

        user_prompt = f"""
        Field Observations:
        {json.dumps(input_data, indent=2)}
        
        Available Candidates (Retrieved from FEMA 306):
        {candidates_text}
        
        Task:
        1. Compare Field Observations (Orientation, Location, Description) against EACH Candidate's Content.
        2. Identify the Candidate whose described failure mechanism BEST matches the observations.
        3. Specifically check if "X-cracking" or "Step-pattern" is mentioned or implied (e.g. "diagonal tension").
        4. Select the BEST matching Candidate ID.
        
        Output Format:
        Selected_ID: <id>
        Score: <float_0_to_1>
        Reasoning: <explanation_citing_specific_quote_from_text>
        Output Format:
        Selected_ID: <id>
        Score: <float_0_to_1>
        Reasoning: <explanation_citing_specific_quote_from_text>
        """

        # --- Hybrid Scoring Calculation ---
        
        # 1. Retrieval Confidence (RC) - Softmax over similarity scores
        # scores from retriever are distances or similarities? 
        # Chroma default for this retriever is likely L2 distance (lower is better) or Cosine Distance.
        # However, checking the output, if they are like 0.8, 1.2 they are distances.
        # If they are like 0.8, 0.7 (descending), they are similarities or inverted distances.
        # We will assume: Probability ~ exp(score) if similarity, or exp(-score) if distance. 
        # Strategy: Normalize via Softmax.
        
        doc_scores = [score for _, score in results]
        
        # Heuristic: if most scores > 1, assume simple distance. 
        # If scores are between 0 and 1, could be cosine distance or similarity.
        # We'll apply softmax directly to the raw scores if we assume they are logits-like or similarities.
        # Ideally we convert to similarity first. 
        
        # Using a safe softmax implementation
        try:
            # Assuming scores are similarities (higher is better). 
            # If they are L2 distances (lower is better), we should negate them.
            # Let's assume similarity for now based on user's "0.81, 0.77" example.
            exp_scores = np.exp(np.array(doc_scores) - np.max(doc_scores)) # shift for stability
            softmax_probs = exp_scores / exp_scores.sum()
        except Exception:
             softmax_probs = [1.0/len(doc_scores)] * len(doc_scores)

        rc_map = {doc.metadata.get("id"): prob for (doc, _), prob in zip(results, softmax_probs)}

        # 2. Cross-Encoder Confidence (ReC)
        rec_map = {}
        if self.cross_encoder:
            try:
                # Pairs of (query, doc_text)
                pairs = [[query, doc.page_content] for doc in candidates]
                ce_scores = self.cross_encoder.predict(pairs)
                
                # Min-Max Normalization
                min_s = ce_scores.min()
                max_s = ce_scores.max()
                if max_s - min_s > 0:
                    norm_ce_scores = (ce_scores - min_s) / (max_s - min_s)
                else:
                    norm_ce_scores = [0.5] * len(ce_scores)
                
                for doc, s in zip(candidates, norm_ce_scores):
                    rec_map[doc.metadata.get("id")] = float(s)
            except Exception as e:
                print(f"CrossEncoder failed: {e}")
        
        # Default ReC if failed or not available
        for doc in candidates:
            did = doc.metadata.get("id")
            if did not in rec_map:
                rec_map[did] = 0.5

        # 4. LLM Generation (Layer 1: Mode Selection)
        response = self.llm_service.generate_response(user_prompt, system_prompt)
        print(f"\n--- DEBUG: LLM Response (Layer 1) ---\n{response}\n-------------------------------------")
        
        # 5. Parse Response for Mode
        selected_id = None
        score = 0.0
        reasoning = response
        
        try:
            import re
            id_match = re.search(r"Selected_ID:\s*(.+)", response)
            if id_match:
                selected_id = id_match.group(1).strip()
            
            score_match = re.search(r"Score:\s*([0-9.]+)", response)
            if score_match:
                score = float(score_match.group(1))
                
            # Clean reasoning
            reasoning = re.sub(r"Selected_ID:.*\n?", "", response)
            reasoning = re.sub(r"Score:.*\n?", "", reasoning).strip()
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")

        # 6. Find the selected doc object
        selected_doc = None
        if selected_id:
            for doc in candidates:
                # fuzzy match ID in case of hallucination or minor format diff
                if selected_id in doc.metadata.get("id", "") or doc.metadata.get("id", "") in selected_id:
                    selected_doc = doc
                    break
        
        # Fallback
        if not selected_doc and score > 0.6: 
             print(f"Warning: LLM selected ID '{selected_id}' not found in candidates.")
             return None

        if not selected_doc:
            return None

        # --- Layer 2: Damage Level Diagnosis & Scoped Reasoning ---
        # Calculate Severity Scope
        width_str = input_data.get("width", "")
        severity, severity_scope = self._calculate_severity(width_str)
        
        print(f"\nCalculated Severity: {severity} (Scope: {severity_scope})")

        l2_system_prompt = (
            "You are a structural engineering expert. You have already identified the Failure Mode. "
            "Now you must explain the specific behavior and failure mechanism at the observed severity level. "
            "Use the calculated 'Severity' context to focus on the relevant part of the 'damage_levels' in the JSON."
        )

        l2_user_prompt = f"""
        Selected Failure Mode Guide:
        {selected_doc.page_content}
        
        Field Observations:
        {json.dumps(input_data, indent=2)}
        
        Calculated Severity (Context): {severity}
        Scope Hint: Focus specifically on the descriptions and criteria for '{severity_scope}' Damage Levels.
        
        Task:
        1. Locate the 'damage_levels' section corresponding to the Scope ({severity_scope}).
        2. Extrapolate the "Scoped Reasoning": Explain WHY the observed cracks (width) are occurring at this specific level. 
           - Quote the specific text describing this level (e.g. "At this level, crushing of units occurs...").
           - Explain what this implies for structural performance (from 'restoration_measures' or 'criteria').
        3. Strict Citation: Identify the Table or Section this Guide represents.
        
        Output Format:
        Damage_Level: <The matched level name, e.g. Moderate>
        Failure_Mode: <The matched failure mode name, e.g. Sliding, Rocking, Cracking, Failure, Buckling, Strut, Yielding, Slip, Crushing, Toe, Tension>
        Scoped_Reasoning: <Detailed explanation derived from the specific level's text>
        Citation: <FEMA 306 Table/Section>
        """
        
        l2_response = self.llm_service.generate_response(l2_user_prompt, l2_system_prompt)
        
        damage_level = "Unknown"
        scoped_reasoning = ""
        citation_text = ""
        
        try:
            dl_match = re.search(r"Damage_Level:\s*(.+)", l2_response)
            if dl_match:
                damage_level = dl_match.group(1).strip()
            
            sr_match = re.search(r"Scoped_Reasoning:\s*(.+?)(?=Citation:|$)", l2_response, re.DOTALL)
            if sr_match:
                scoped_reasoning = sr_match.group(1).strip()
                
            cit_match = re.search(r"Citation:\s*(.+)", l2_response)
            if cit_match:
                citation_text = cit_match.group(1).strip()
        except Exception as e:
            print(f"Error parsing Layer 2 response: {e}")

        citations = [
            Citation(
                section_id=selected_doc.metadata.get("id"),
                title=selected_doc.metadata.get("title"),
                text_snippet=f"Citation: {citation_text}\n\n{selected_doc.page_content[:500]}...",
            )
        ]
        
        # Combine Step 1 Identification with Step 2 specific context
        final_reasoning = (
            f"**Mechanism & Severity Analysis:**\n{scoped_reasoning}"
        )

        # Compute Final Hybrid Score for the selected doc
        # Formula: 0.5 * LLM + 0.3 * RC + 0.2 * ReC
        rc_val = 0.0
        rec_val = 0.0
        
        if selected_doc:
            sel_id = selected_doc.metadata.get("id")
            rc_val = rc_map.get(sel_id, 0.0)
            rec_val = rec_map.get(sel_id, 0.0)
            
        final_hybrid_score = (0.5 * score) + (0.3 * rc_val) + (0.2 * rec_val)
        final_hybrid_score = round(final_hybrid_score, 4)

        return DiagnosisResult(
            failure_mode=FailureMode(
                id=selected_doc.metadata.get("id"),
                name=selected_doc.metadata.get("title"),
                material=selected_doc.metadata.get("material", "URM"),
                type=selected_doc.metadata.get("mode_type", "General"),
                description=selected_doc.page_content
            ),
            confidence_score=score,
            reasoning=final_reasoning,
            citations=citations,
            damage_level=damage_level,
            severity=severity,
            # Hybrid Scores
            hybrid_score=final_hybrid_score,
            llm_confidence=score, # LLM_SC
            retrieval_confidence=rc_val, # RC
            cross_encoder_confidence=rec_val # ReC
        )
