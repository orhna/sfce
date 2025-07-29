#!/usr/bin/env python3
"""
Main script for Short Form Content Extraction from Twitch streams.

Usage:
    python main.py <vod_url> [interval_seconds] [top_highlights] [pre_padding_seconds] [--reset-db]

Example:
    python main.py https://www.twitch.tv/videos/2518537772 5 10 5 --reset-db
"""

import sys
import argparse
from utils.utils import comprehensive_multi_modal_workflow
from utils.db import setup_database, reset_database


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='Short Form Content Extraction from Twitch streams',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py https://www.twitch.tv/videos/2518537772
  python main.py https://www.twitch.tv/videos/2518537772 5 10 5
  python main.py https://www.twitch.tv/videos/2518537772 10 15 3 --reset-db
        """
    )
    
    # Required argument
    parser.add_argument('vod_url', help='Twitch VOD URL (required)')
    
    # Optional positional arguments
    parser.add_argument('interval_seconds', nargs='?', type=int, default=5,
                       help='Analysis interval in seconds (default: 5)')
    parser.add_argument('top_highlights', nargs='?', type=int, default=10,
                       help='Number of highlights to extract (default: 10)')
    parser.add_argument('pre_padding_seconds', nargs='?', type=int, default=5,
                       help='Padding before highlights in seconds (default: 5)')
    
    # Optional flag arguments
    parser.add_argument('--reset-db', action='store_true',
                       help='Reset the database before processing (deletes all entries)')
    
    args = parser.parse_args()
    
    # Handle database reset
    if args.reset_db:
        print("🗑️  Database reset requested...")
        try:
            reset_database()
        except Exception as e:
            print(f"❌ Failed to reset database: {e}")
            print("Continuing with workflow...")
    
    # Setup database
    print("🔧 Setting up database...")
    setup_database()
    
    # Run comprehensive workflow
    print(f"🚀 Starting comprehensive analysis of: {args.vod_url}")
    print(f"   Interval: {args.interval_seconds}s | Top highlights: {args.top_highlights} | Padding: {args.pre_padding_seconds}s")
    if args.reset_db:
        print("   🗑️  Database was reset before processing")
    
    comprehensive_multi_modal_workflow(
        vod_url=args.vod_url,
        interval_seconds=args.interval_seconds,
        top_highlights=args.top_highlights,
        pre_padding_seconds=args.pre_padding_seconds
    )

if __name__ == "__main__":
    main() 