#!/usr/bin/env python3
"""
JSesh Integration Module for Digital Paleography

This module integrates JSesh mappings with existing Unicode mappings to provide
hieroglyphic notation including:
- Gardiner codes (A1, D4, etc.)
- Unicode symbols (𓀀, 𓁹, etc.)  
- JSesh transliteration codes (A1, D4, etc.)
- Full paleographic descriptions

"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class JSeshIntegrator:
    def __init__(self):
        """Initialize the JSesh integrator with all mapping data"""
        self.project_root = Path(__file__).parent.parent
        self.jsesh_mappings = {}
        self.unicode_mappings = {}
        self.gardiner_descriptions = {}
        
        # Load all mapping data
        self.load_jsesh_mappings()
        self.load_unicode_mappings()
        self.load_gardiner_descriptions()
        
        # Create reverse mappings for efficiency
        self.create_reverse_mappings()
    
    def load_jsesh_mappings(self):
        """Load Unicode → JSesh mappings"""
        jsesh_file = self.project_root / "data/ "unicode_jsesh_mappings.json"
        try:
            with open(jsesh_file, 'r', encoding='utf-8') as f:
                self.jsesh_mappings = json.load(f)
            print(f"Loaded {len(self.jsesh_mappings)} JSesh mappings")
        except Exception as e:
            print(f"Could not load JSesh mappings: {e}")
            self.jsesh_mappings = {}
    
    def load_unicode_mappings(self):
        """Load Gardiner → Unicode mappings"""
        unicode_file = (self.project_root / "data/ "annotations/ 
                        "gardiner_unicode_mapping.json")
        try:
            with open(unicode_file, 'r', encoding='utf-8') as f:
                self.unicode_mappings = json.load(f)
            print(f"Loaded {len(self.unicode_mappings)} Gardiner → Unicode mappings")
        except Exception as e:
            print(f"Could not load Unicode mappings: {e}")
            self.unicode_mappings = {}
    
    def load_gardiner_descriptions(self):
        """Load Gardiner code descriptions"""
        descriptions_file = (self.project_root / "data/ "annotations/ 
                             "gardiner_descriptions.json")
        try:
            with open(descriptions_file, 'r', encoding='utf-8') as f:
                self.gardiner_descriptions = json.load(f)
            print(f"Loaded {len(self.gardiner_descriptions)} Gardiner descriptions")
        except Exception as e:
            print(fCould not load Gardiner descriptions: {e}")
            self.gardiner_descriptions = {}
    
    def create_reverse_mappings(self):
        """Create reverse mappings for efficient lookup"""
        # Unicode → Gardiner mapping
        self.unicode_to_gardiner = {}
        for gardiner_code, data in self.unicode_mappings.items():
            unicode_codes = data.get('unicode_codes', [])
            for unicode_code in unicode_codes:
                if unicode_code not in self.unicode_to_gardiner:
                    self.unicode_to_gardiner[unicode_code] = []
                self.unicode_to_gardiner[unicode_code].append(gardiner_code)
        
        # JSesh → Unicode mapping
        self.jsesh_to_unicode = {}
        for unicode_code, jsesh_code in self.jsesh_mappings.items():
            if jsesh_code not in self.jsesh_to_unicode:
                self.jsesh_to_unicode[jsesh_code] = []
            self.jsesh_to_unicode[jsesh_code].append(unicode_code)
    
    def find_fallback_code(self, gardiner_code: str) -> Optional[str]:
        """Find fallback Gardiner code for variants (e.g., A1B -> A1)"""
        if gardiner_code in self.unicode_mappings:
            return gardiner_code  # Exact match found
        
        # Try removing suffix letters (A1B -> A1, D4C -> D4)
        import re
        
        # Extract base code (letters + numbers)
        match = re.match(r'^([A-Z]+\d+)', gardiner_code)
        if match:
            base_code = match.group(1)
            if base_code in self.unicode_mappings:
                return base_code
        
        # Try removing just the last character (A1a -> A1)
        if len(gardiner_code) > 2:
            potential_base = gardiner_code[:-1]
            if potential_base in self.unicode_mappings:
                return potential_base
        
        return None
    
    def get_unicode_symbol(self, gardiner_code: str) -> Optional[str]:
        """Get Unicode symbol for a Gardiner code with fallback support"""
        # Try to find the code or its fallback
        actual_code = self.find_fallback_code(gardiner_code)
        if not actual_code:
            return None
        
        unicode_codes = self.unicode_mappings[actual_code].get('unicode_codes', [])
        if not unicode_codes:
            return None
        
        # Get the first valid Unicode code
        for code in unicode_codes:
            if code.startswith('U+'):
                try:
                    unicode_int = int(code[2:], 16)
                    return chr(unicode_int)
                except (ValueError, OverflowError):
                    continue
        
        return None
    
    def get_jsesh_code(self, gardiner_code: str) -> Optional[str]:
        """Get JSesh code for a Gardiner code with fallback support"""
        # Try to find the code or its fallback
        actual_code = self.find_fallback_code(gardiner_code)
        if not actual_code:
            return None
        
        unicode_codes = self.unicode_mappings[actual_code].get('unicode_codes', [])
        
        # Try each Unicode code to find a JSesh mapping
        for unicode_code in unicode_codes:
            if unicode_code in self.jsesh_mappings:
                return self.jsesh_mappings[unicode_code]
        
        return None
    
    def get_complete_notation(self, gardiner_code: str) -> Dict:
        """Get complete notation for a Gardiner code including all available data with fallback"""
        notation = {
            'gardiner_code': gardiner_code,
            'unicode_symbol': None,
            'unicode_codes': [],
            'jsesh_code': None,
            'description': None,
            'available_notations': [],
            'fallback_used': None
        }
        
        # Find the actual code to use (original or fallback)
        actual_code = self.find_fallback_code(gardiner_code)
        
        if actual_code and actual_code != gardiner_code:
            notation['fallback_used'] = actual_code
        
        # Get Unicode information
        if actual_code and actual_code in self.unicode_mappings:
            unicode_data = self.unicode_mappings[actual_code]
            notation['unicode_codes'] = unicode_data.get('unicode_codes', [])
            notation['unicode_symbol'] = self.get_unicode_symbol(gardiner_code)
            if notation['unicode_symbol']:
                notation['available_notations'].append('unicode')
        
        # Get JSesh code
        notation['jsesh_code'] = self.get_jsesh_code(gardiner_code)
        if notation['jsesh_code']:
            notation['available_notations'].append('jsesh')
        
        # Get description - try original first, then fallback
        notation['description'] = self.gardiner_descriptions.get(
            gardiner_code, 
            self.gardiner_descriptions.get(
                actual_code if actual_code else gardiner_code, 
                f"Hieroglyph {gardiner_code}"
            )
        )
        
        # Add Gardiner code as always available
        notation['available_notations'].append('gardiner')
        
        return notation
    
    def get_enhanced_crop_data(self, crop_data: Dict) -> Dict:
        """Enhance existing crop data with complete notation information"""
        gardiner_code = crop_data.get('gardiner_code', 'Unknown')
        
        # Get complete notation
        notation = self.get_complete_notation(gardiner_code)
        
        # Enhance the crop data
        enhanced_crop = crop_data.copy()
        enhanced_crop.update({
            'unicode_symbol': notation['unicode_symbol'],
            'unicode_codes': notation['unicode_codes'],
            'jsesh_code': notation['jsesh_code'],
            'description': notation['description'],
            'available_notations': notation['available_notations'],
            'fallback_used': notation['fallback_used'],  # Include fallback information
            'enhanced': True  # Flag to indicate this has been enhanced
        })
        
        return enhanced_crop
    
    def create_notation_summary(self) -> Dict:
        """Create a summary of all available notations"""
        summary = {
            'total_gardiner_codes': len(self.unicode_mappings),
            'total_unicode_mappings': len(self.jsesh_mappings),
            'total_jsesh_mappings': len(self.jsesh_mappings),
            'coverage_analysis': {},
            'notation_types': {
                'gardiner_only': 0,
                'gardiner_unicode': 0,
                'gardiner_jsesh': 0,
                'complete_notation': 0  # All three: Gardiner + Unicode + JSesh
            }
        }
        
        # Analyze coverage
        for gardiner_code in self.unicode_mappings.keys():
            notation = self.get_complete_notation(gardiner_code)
            notations = set(notation['available_notations'])
            
            if notations == {'gardiner'}:
                summary['notation_types']['gardiner_only'] += 1
            elif 'unicode' in notations and 'jsesh' not in notations:
                summary['notation_types']['gardiner_unicode'] += 1
            elif 'jsesh' in notations and 'unicode' not in notations:
                summary['notation_types']['gardiner_jsesh'] += 1
            elif 'unicode' in notations and 'jsesh' in notations:
                summary['notation_types']['complete_notation'] += 1
        
        # Calculate enhancement potential
        complete_coverage = summary['notation_types']['complete_notation']
        total_codes = summary['total_gardiner_codes']
        summary['enhancement_potential'] = {
            'complete_coverage_percent': (complete_coverage / total_codes) * 100 if total_codes > 0 else 0,
            'enhanced_codes': complete_coverage,
            'total_codes': total_codes
        }
        
        return summary
    
    def export_enhanced_mappings(self, output_path: Optional[Path] = None) -> Path:
        """Export complete enhanced mappings combining all notation systems"""
        if output_path is None:
            output_path = self.project_root / "data/ "enhanced_hieroglyph_mappings.json"
        
        enhanced_mappings = {}
        
        # Process all known Gardiner codes
        all_codes = set(self.unicode_mappings.keys())
        
        for gardiner_code in sorted(all_codes):
            enhanced_mappings[gardiner_code] = self.get_complete_notation(gardiner_code)
        
        # Save enhanced mappings
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_mappings, f, indent=2, ensure_ascii=False)
        
        print(f"Enhanced mappings exported to: {output_path}")
        return output_path
    
    def demonstrate_enhancement(self, sample_codes: List[str] = None) -> None:
        """Demonstrate the enhancement capabilities with sample codes"""
        if sample_codes is None:
            # Use common codes from your training data
            sample_codes = ['A1', 'D4', 'D21', 'G1', 'G43', 'M17', 'N35', 'X1', 'Y1', 'Z1']
        
        print("JSesh Integration Demonstration")
        print("=* 60)
        
        for code in sample_codes:
            notation = self.get_complete_notation(code)
            
            print(f"\n{code} - {notation['description']}")
            print(fGardiner: {code}")
            
            if notation['unicode_symbol']:
                unicode_codes_str = ', '.join(notation['unicode_codes'])
                print(fUnicode:  {notation['unicode_symbol']} ({unicode_codes_str})")
            else:
                print(fUnicode:  Not available")
            
            if notation['jsesh_code']:
                print(fJSesh:    {notation['jsesh_code']}")
            else:
                print(fJSesh:    Not available")
            
            print(fAvailable: {', '.join(notation['available_notations'])}")


def main():
    """Demonstrate the JSesh integration capabilities"""
    print("JSesh Integration For Digital Paleography")
    print("=* 50)
    
    # Initialize integrator
    integrator = JSeshIntegrator()
    
    # Show coverage summary
    summary = integrator.create_notation_summary()
    print(f"\Notation Coverage Summary:")
    print(f"Total Gardiner codes: {summary['total_gardiner_codes']}")
    print(f"Complete notation coverage: {summary['notation_types']['complete_notation']} codes")
    print(f"Coverage percentage: {summary['enhancement_potential']['complete_coverage_percent']:.1f}%")
    
    # Demonstrate with sample codes
    integrator.demonstrate_enhancement()
    
    # Export enhanced mappings
    output_file = integrator.export_enhanced_mappings()
    print(f"\nEnhanced mappings saved to: {output_file}")
    
    print(f"\Integration ready!")
    print("Your digital paleography tool can now provide:")
    print("• Gardiner codes (A1, D4, etc.)")
    print("• Unicode symbols (𓀀, 𓁹, etc.)")
    print("• JSesh transliteration (A1, D4, etc.)")
    print("• Complete scholarly notation!")


if __name__ == "__main__":
    main()
