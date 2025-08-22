"""
Brand Detection Utility

This module provides sophisticated brand detection capabilities with support
for variations, context analysis, and ranking detection (Stage 2 preparation).
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from src.workflow.state import BrandDetectionResult
from src.config.settings import get_settings

@dataclass
class MatchContext:
    """Context information around a brand match."""
    text: str
    start_pos: int
    end_pos: int
    surrounding_context: str
    sentence: str

class BrandDetector:
    """Advanced brand detection with context analysis and ranking detection."""
    
    def __init__(self, config=None):
        """Initialize the brand detector with configuration."""
        self.config = config or get_settings().brand
        self.ranking_keywords = get_settings().ranking_keywords
        
        # Prepare brand variations for efficient matching
        self.brand_patterns = self._prepare_brand_patterns()
        self.ranking_patterns = self._prepare_ranking_patterns()
    
    def _prepare_brand_patterns(self) -> List[re.Pattern]:
        """Prepare regex patterns for brand matching."""
        patterns = []
        
        for variation in self.config.brand_variations:
            # Escape special regex characters
            escaped = re.escape(variation)
            
            if self.config.partial_match:
                # Allow partial matches within word boundaries
                pattern = rf'\b{escaped}\b'
            else:
                # Exact match only
                pattern = rf'^{escaped}$'
            
            flags = 0 if self.config.case_sensitive else re.IGNORECASE
            patterns.append(re.compile(pattern, flags))
        
        return patterns
    
    def _prepare_ranking_patterns(self) -> List[re.Pattern]:
        """Prepare regex patterns for ranking detection (Stage 2)."""
        patterns = []
        
        # Ordinal numbers (1st, 2nd, 3rd, etc.)
        ordinal_pattern = r'\b(\d+)(?:st|nd|rd|th)\b'
        patterns.append(re.compile(ordinal_pattern, re.IGNORECASE))
        
        # Ranking keywords
        for keyword in self.ranking_keywords:
            pattern = rf'\b{re.escape(keyword)}\b'
            patterns.append(re.compile(pattern, re.IGNORECASE))
        
        # Number-based rankings (#1, #2, etc.)
        number_ranking = r'#(\d+)'
        patterns.append(re.compile(number_ranking))
        
        return patterns
    
    def detect_brand(self, text: str, include_ranking: bool = False) -> BrandDetectionResult:
        """
        Detect brand mentions in the given text.
        
        Args:
            text: Text to analyze
            include_ranking: Whether to detect ranking information (Stage 2)
            
        Returns:
            BrandDetectionResult with detection information
        """
        if not text:
            return BrandDetectionResult(found=False, confidence=0.0)
        
        matches = self._find_brand_matches(text)
        
        if not matches:
            return BrandDetectionResult(found=False, confidence=0.0)
        
        # Calculate confidence based on number and quality of matches
        confidence = self._calculate_confidence(matches, text)
        
        # Extract match information
        match_texts = [match.text for match in matches]
        best_context = max(matches, key=lambda m: len(m.surrounding_context)).surrounding_context
        
        result = BrandDetectionResult(
            found=True,
            confidence=confidence,
            matches=match_texts,
            context=best_context
        )
        
        # Add ranking detection if requested (Stage 2 preparation)
        if include_ranking:
            ranking_info = self._detect_ranking(text, matches)
            result.ranking_position = ranking_info.get('position')
            result.ranking_context = ranking_info.get('context')
        
        return result
    
    def _find_brand_matches(self, text: str) -> List[MatchContext]:
        """Find all brand matches in the text."""
        matches = []
        
        for pattern in self.brand_patterns:
            for match in pattern.finditer(text):
                context = self._extract_context(text, match)
                matches.append(context)
        
        return matches
    
    def _extract_context(self, text: str, match: re.Match) -> MatchContext:
        """Extract context around a brand match."""
        start, end = match.span()
        
        # Get surrounding context (50 characters before and after)
        context_start = max(0, start - 50)
        context_end = min(len(text), end + 50)
        surrounding_context = text[context_start:context_end]
        
        # Find the sentence containing the match
        sentence = self._find_containing_sentence(text, start, end)
        
        return MatchContext(
            text=match.group(),
            start_pos=start,
            end_pos=end,
            surrounding_context=surrounding_context,
            sentence=sentence
        )
    
    def _find_containing_sentence(self, text: str, start: int, end: int) -> str:
        """Find the sentence that contains the match."""
        # Simple sentence boundary detection
        sentence_endings = ['.', '!', '?', '\n']
        
        # Find sentence start
        sentence_start = 0
        for i in range(start - 1, -1, -1):
            if text[i] in sentence_endings:
                sentence_start = i + 1
                break
        
        # Find sentence end
        sentence_end = len(text)
        for i in range(end, len(text)):
            if text[i] in sentence_endings:
                sentence_end = i + 1
                break
        
        return text[sentence_start:sentence_end].strip()
    
    def _calculate_confidence(self, matches: List[MatchContext], text: str) -> float:
        """Calculate confidence score based on matches and context."""
        if not matches:
            return 0.0
        
        base_confidence = 0.6  # Base confidence for any match
        
        # Bonus for multiple matches
        multiple_match_bonus = min(0.2, len(matches) * 0.05)
        
        # Bonus for exact brand name matches
        exact_match_bonus = 0.0
        for match in matches:
            if match.text.lower() == self.config.target_brand.lower():
                exact_match_bonus = 0.1
                break
        
        # Context quality bonus
        context_bonus = self._calculate_context_bonus(matches)
        
        confidence = base_confidence + multiple_match_bonus + exact_match_bonus + context_bonus
        return min(1.0, confidence)  # Cap at 1.0
    
    def _calculate_context_bonus(self, matches: List[MatchContext]) -> float:
        """Calculate bonus based on context quality."""
        positive_keywords = [
            'best', 'top', 'leading', 'excellent', 'great', 'recommend',
            'powerful', 'innovative', 'solution', 'platform', 'tool'
        ]
        
        negative_keywords = [
            'worst', 'bad', 'poor', 'avoid', 'terrible', 'disappointing'
        ]
        
        context_score = 0.0
        
        for match in matches:
            context_lower = match.surrounding_context.lower()
            
            # Positive context
            positive_count = sum(1 for keyword in positive_keywords if keyword in context_lower)
            context_score += positive_count * 0.02
            
            # Negative context (reduces confidence)
            negative_count = sum(1 for keyword in negative_keywords if keyword in context_lower)
            context_score -= negative_count * 0.05
        
        return max(-0.1, min(0.1, context_score))  # Cap between -0.1 and 0.1
    
    def _detect_ranking(self, text: str, brand_matches: List[MatchContext]) -> Dict[str, Optional[any]]:
        """
        Detect ranking information near brand mentions (Stage 2 preparation).
        
        Returns:
            Dictionary with 'position' and 'context' keys
        """
        if not brand_matches:
            return {'position': None, 'context': None}
        
        ranking_info = {'position': None, 'context': None}
        
        for brand_match in brand_matches:
            # Look for ranking indicators near the brand mention
            context_text = brand_match.sentence
            
            # Check for ordinal rankings
            for pattern in self.ranking_patterns:
                for match in pattern.finditer(context_text):
                    # Extract position if it's a number
                    if match.group().isdigit() or match.group().replace('#', '').isdigit():
                        try:
                            position = int(match.group().replace('#', '').replace('st', '').replace('nd', '').replace('rd', '').replace('th', ''))
                            if 1 <= position <= 100:  # Reasonable ranking range
                                ranking_info['position'] = position
                                ranking_info['context'] = context_text
                                return ranking_info
                        except ValueError:
                            continue
                    
                    # Check for keyword-based rankings
                    if any(keyword in match.group().lower() for keyword in ['first', 'top', 'best', 'leading', 'number one']):
                        ranking_info['position'] = 1
                        ranking_info['context'] = context_text
                        return ranking_info
        
        return ranking_info
    
    def batch_detect(self, texts: List[str], include_ranking: bool = False) -> List[BrandDetectionResult]:
        """Batch process multiple texts for brand detection."""
        return [self.detect_brand(text, include_ranking) for text in texts]
    
    def get_detection_summary(self, results: List[BrandDetectionResult]) -> Dict[str, any]:
        """Generate a summary of detection results."""
        total = len(results)
        found_count = sum(1 for r in results if r.found)
        
        if found_count == 0:
            avg_confidence = 0.0
        else:
            avg_confidence = sum(r.confidence for r in results if r.found) / found_count
        
        return {
            'total_analyzed': total,
            'brand_mentions_found': found_count,
            'detection_rate': found_count / total if total > 0 else 0.0,
            'average_confidence': avg_confidence,
            'unique_matches': list(set(match for r in results for match in r.matches))
        }

# Utility functions for easy usage
def detect_brand_in_text(text: str, config=None) -> BrandDetectionResult:
    """Simple function to detect brand in a single text."""
    detector = BrandDetector(config)
    return detector.detect_brand(text)

def detect_brand_batch(texts: List[str], config=None) -> List[BrandDetectionResult]:
    """Simple function to detect brand in multiple texts."""
    detector = BrandDetector(config)
    return detector.batch_detect(texts)