#!/usr/bin/env python3
"""
Generate 2D PDF cutting templates from pantry shelf configuration.

Usage:
    python scripts/render_2d.py configs/pantry_0000.json
    python scripts/render_2d.py configs/pantry_0000.json --output output/custom/
    python scripts/render_2d.py configs/pantry_0000.json --no-overview
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import config
import pdf_generator

ShelfConfig = config.ShelfConfig
PDFGenerator = pdf_generator.PDFGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Generate 2D PDF cutting templates for pantry shelves'
    )
    parser.add_argument(
        'config',
        type=Path,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output directory (default: output/cutting_templates/)'
    )
    parser.add_argument(
        '--no-overview',
        action='store_true',
        help='Skip overview pages showing all shelves per level'
    )

    args = parser.parse_args()

    # Validate input file
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    # Load configuration
    print(f"Loading configuration from {args.config}")
    try:
        config = ShelfConfig.from_file(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_dir = args.output
    else:
        output_dir = Path('output/cutting_templates')

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename based on config version
    output_file = output_dir / f'pantry_{config.version}_cutting_templates.pdf'

    # Generate PDF
    print(f"\nGenerating PDF cutting templates...")
    print(f"  Configuration: {config.version}")
    print(f"  Shelves: {len(config.shelves)} sections")
    print(f"  Output: {output_file}")

    try:
        generator = PDFGenerator(config)
        generator.generate_pdf(output_file, include_overview=not args.no_overview)
        print("\nSuccess! PDF generated.")
        print(f"\nOpen with: open {output_file}")
    except Exception as e:
        print(f"\nError generating PDF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
