#!/usr/bin/env python3
"""
Examine the compiled pipeline JSON to find the UTF-8 issue at position 950-951.
This will identify the exact problematic content in the pipeline file.
"""

import json
import os
import sys
from kfp.v2 import compiler
from promotion_constants import PIPELINE_FILE


def analyze_compiled_pipeline():
    """Compile pipeline and analyze the resulting JSON for UTF-8 issues."""
    print("🔍 Analyzing compiled pipeline JSON for UTF-8 issues")
    print("=" * 60)
    
    # First, compile the pipeline
    try:
        from promotion import model_promotion_pipeline
        
        print("📝 Compiling pipeline...")
        compiler.Compiler().compile(
            pipeline_func=model_promotion_pipeline,
            package_path=PIPELINE_FILE
        )
        print(f"✅ Pipeline compiled to: {PIPELINE_FILE}")
        
    except Exception as e:
        print(f"❌ Pipeline compilation failed: {e}")
        return
    
    # Read and analyze the compiled JSON
    if not os.path.exists(PIPELINE_FILE):
        print(f"❌ Pipeline file not found: {PIPELINE_FILE}")
        return
    
    print(f"\n📖 Reading pipeline file: {PIPELINE_FILE}")
    
    try:
        # Read as binary first to check raw content
        with open(PIPELINE_FILE, 'rb') as f:
            raw_content = f.read()
        
        print(f"📊 File size: {len(raw_content)} bytes")
        
        # Check around position 950-951 (the error location)
        error_position = 950
        start_pos = max(0, error_position - 50)
        end_pos = min(len(raw_content), error_position + 50)
        
        print(f"\n🎯 Content around position {error_position} (±50 characters):")
        print(f"Position range: {start_pos} to {end_pos}")
        print("-" * 40)
        
        # Show raw bytes
        problematic_bytes = raw_content[start_pos:end_pos]
        print("Raw bytes:")
        for i, byte in enumerate(problematic_bytes):
            pos = start_pos + i
            if pos == error_position or pos == error_position + 1:
                print(f">>> {pos:4d}: 0x{byte:02X} ({chr(byte) if 32 <= byte <= 126 else '?'})")
            else:
                print(f"    {pos:4d}: 0x{byte:02X} ({chr(byte) if 32 <= byte <= 126 else '?'})")
        
        # Try to decode as UTF-8 and identify problematic characters
        print(f"\n🔤 UTF-8 decoding analysis:")
        try:
            text_content = raw_content.decode('utf-8')
            problematic_text = text_content[start_pos:end_pos]
            print(f"Decoded text: {repr(problematic_text)}")
            
            # Character by character analysis
            print("\nCharacter analysis:")
            for i, char in enumerate(problematic_text):
                pos = start_pos + i
                try:
                    char_code = ord(char)
                    is_problem = pos == error_position or pos == error_position + 1
                    marker = ">>> " if is_problem else "    "
                    print(f"{marker}{pos:4d}: '{char}' (U+{char_code:04X})")
                except Exception as e:
                    print(f">>> {pos:4d}: ERROR - {e}")
                    
        except UnicodeDecodeError as e:
            print(f"❌ UTF-8 decoding failed: {e}")
            print(f"Error at byte position: {e.start}")
            
            # Show the problematic bytes
            if e.start < len(raw_content):
                problem_byte = raw_content[e.start]
                print(f"Problematic byte: 0x{problem_byte:02X}")
        
        # Try to load as JSON to see where it fails
        print(f"\n📋 JSON parsing analysis:")
        try:
            with open(PIPELINE_FILE, 'r', encoding='utf-8') as f:
                json_content = json.load(f)
            print("✅ JSON parsing successful")
            
            # Try to re-serialize to find the issue
            try:
                json_str = json.dumps(json_content)
                print("✅ JSON serialization successful")
            except Exception as e:
                print(f"❌ JSON serialization failed: {e}")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"Error at position: {e.pos}")
        except UnicodeDecodeError as e:
            print(f"❌ UTF-8 decoding during JSON read failed: {e}")
            
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")


def find_problematic_content_in_components():
    """Check if the issue is in the component definitions themselves."""
    print(f"\n🔍 Analyzing component definitions for UTF-8 issues")
    print("=" * 60)
    
    # Check component docstrings and code
    component_texts = [
        ("get_source_model docstring", """Get model information from source registry."""),
        ("validate_model_for_promotion docstring", """Validate model meets promotion criteria."""),
        ("promote_model_to_target docstring", """Promote model to target registry."""),
        ("send_promotion_notification docstring", """Send notification about promotion completion."""),
        ("pipeline docstring", """Enhanced model promotion pipeline with preparation steps."""),
    ]
    
    for name, text in component_texts:
        try:
            # Test UTF-8 encoding
            encoded = text.encode('utf-8')
            json.dumps(text)  # Test JSON serialization
            print(f"✅ {name}: OK")
        except Exception as e:
            print(f"❌ {name}: {e}")
            print(f"   Content: {repr(text)}")


def create_minimal_test_pipeline():
    """Create a minimal pipeline to isolate the UTF-8 issue."""
    print(f"\n🧪 Creating minimal test pipeline")
    print("=" * 60)
    
    from kfp.v2 import dsl
    from kfp.v2.dsl import component
    
    @component(base_image="python:3.9")
    def minimal_test_component(message: str) -> str:
        """Minimal test component."""
        return f"Processed: {message}"
    
    @dsl.pipeline(name="minimal-test-pipeline")
    def minimal_test_pipeline(input_message: str = "hello"):
        """Minimal test pipeline."""
        task = minimal_test_component(message=input_message)
    
    try:
        test_file = "minimal_test_pipeline.json"
        compiler.Compiler().compile(
            pipeline_func=minimal_test_pipeline,
            package_path=test_file
        )
        print(f"✅ Minimal pipeline compiled successfully: {test_file}")
        
        # Check if minimal pipeline has UTF-8 issues
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
            json.loads(content)  # Test JSON parsing
        print("✅ Minimal pipeline JSON is valid UTF-8")
        
        # Clean up
        os.unlink(test_file)
        
    except Exception as e:
        print(f"❌ Minimal pipeline failed: {e}")


def main():
    """Run all diagnostic checks."""
    print("🚀 Pipeline JSON UTF-8 Diagnostic")
    print("=" * 60)
    
    # Check compiled pipeline
    analyze_compiled_pipeline()
    
    # Check component definitions
    find_problematic_content_in_components()
    
    # Test minimal pipeline
    create_minimal_test_pipeline()
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("Based on the analysis above:")
    print("1. If specific characters are identified at position 950-951, replace them")
    print("2. If the issue is in component docstrings, clean them up")
    print("3. If the minimal pipeline works, the issue is in your specific components")
    print("4. Consider using the UTF-8 safe pipeline version as a permanent fix")


if __name__ == "__main__":
    main()