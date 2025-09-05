#!/bin/bash
#
# Clean macOS Metadata Files - PapyrusVision
# ==========================================
# 
# This script removes macOS metadata files that can clutter the project
# and cause issues when sharing across different platforms.
#
# Usage: ./clean_macos_metadata.sh
#

echo "🧹 Cleaning macOS metadata files in PapyrusVision..."

# Count files before cleaning
before_count=$(find . -name "._*" -o -name ".DS_Store" -o -name "__MACOSX" -o -name ".AppleDouble" -o -name ".LSOverride" | wc -l | tr -d ' ')

if [ "$before_count" -eq 0 ]; then
    echo "✅ No macOS metadata files found - PapyrusVision is already clean!"
    exit 0
fi

echo "📊 Found $before_count metadata files to remove:"

# Show what will be removed
find . \( -name "._*" -o -name ".DS_Store" -o -name "__MACOSX" -o -name ".AppleDouble" -o -name ".LSOverride" \) -print

echo ""
read -p "🤔 Remove all these files? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Remove the files
    find . \( -name "._*" -o -name ".DS_Store" -o -name "__MACOSX" -o -name ".AppleDouble" -o -name ".LSOverride" \) -delete
    
    # Count files after cleaning
    after_count=$(find . -name "._*" -o -name ".DS_Store" -o -name "__MACOSX" -o -name ".AppleDouble" -o -name ".LSOverride" | wc -l | tr -d ' ')
    removed_count=$((before_count - after_count))
    
    echo "✅ Successfully removed $removed_count metadata files!"
    
    if [ "$after_count" -eq 0 ]; then
        echo "🎉 PapyrusVision is now clean of macOS metadata files!"
    else
        echo "⚠️  $after_count files could not be removed (check permissions)"
    fi
    
    echo ""
    echo "💡 Tip: These files are already ignored by .gitignore, so they won't"
    echo "   be committed to git in the future."
    echo ""
    echo "🔬 For AI-powered hieroglyph detection, run:"
    echo "   ./run_with_env.sh streamlit run apps/unified_papyrus_app.py"
    
else
    echo "❌ Cleanup cancelled"
fi
