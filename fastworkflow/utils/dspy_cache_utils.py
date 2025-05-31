#!/usr/bin/env python3
"""
DSPy Cache Management Utilities

This script provides easy commands to manage DSPy LLM caches.
"""
import os
import shutil
import argparse
from pathlib import Path
from typing import Optional

import dspy


def clear_dspy_cache_completely():
    """Clear DSPy cache by disabling all caching mechanisms."""
    print("🗑️  Clearing DSPy cache (disabling all cache mechanisms)...")
    
    # Disable all DSPy caching
    dspy.configure_cache(
        enable_disk_cache=False,
        enable_memory_cache=False,
        enable_litellm_cache=False
    )
    
    print("✅ DSPy cache cleared - all future LLM calls will be fresh")


def clear_dspy_disk_cache(cache_dir: Optional[str] = None):
    """Clear DSPy disk cache files."""
    # Default DSPy cache directory (usually ~/.cache/dspy or similar)
    if cache_dir is None:
        # Try to find the default cache directory
        possible_dirs = [
            Path.home() / ".cache" / "dspy",
            Path.home() / ".dspy_cache",
            Path.cwd() / ".dspy_cache",
        ]
        
        for cache_path in possible_dirs:
            if cache_path.exists():
                cache_dir = str(cache_path)
                break
    
    if cache_dir and Path(cache_dir).exists():
        print(f"🗑️  Clearing DSPy disk cache directory: {cache_dir}")
        try:
            shutil.rmtree(cache_dir)
            print(f"✅ Successfully cleared disk cache: {cache_dir}")
        except Exception as e:
            print(f"❌ Error clearing disk cache: {e}")
    else:
        print("ℹ️  No DSPy disk cache directory found to clear")


def reset_dspy_cache_settings():
    """Reset DSPy cache to default settings."""
    print("🔄 Resetting DSPy cache to default settings...")
    
    dspy.configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        enable_litellm_cache=False
    )
    
    print("✅ DSPy cache reset to defaults")


def show_cache_status():
    """Show current DSPy cache configuration."""
    print("📊 Current DSPy Cache Status:")

    try:
        # Access DSPy cache settings if available
        if hasattr(dspy, 'cache') and dspy.cache:
            cache = dspy.cache
            print(f"  • Disk cache enabled: {getattr(cache, 'enable_disk_cache', 'Unknown')}")
            print(f"  • Memory cache enabled: {getattr(cache, 'enable_memory_cache', 'Unknown')}")
            print(f"  • Cache directory: {getattr(cache, 'disk_cache_dir', 'Unknown')}")
        else:
            print("  • Cache object not found - may be disabled")

        # Check if LiteLLM cache is enabled
        try:
            import litellm
            if hasattr(litellm, 'cache') and litellm.cache:
                print("  • LiteLLM cache enabled: True")
            else:
                print("  • LiteLLM cache enabled: False")
        except ImportError:
            print("  • LiteLLM not available")

    except Exception as e:
        print(f"  • Error checking cache status: {e}")


def main():
    parser = argparse.ArgumentParser(description="DSPy Cache Management Utilities")
    parser.add_argument("command", choices=[
        "clear", "clear-disk", "reset", "status"
    ], help="Cache operation to perform")
    parser.add_argument("--cache-dir", help="Custom cache directory path")
    
    args = parser.parse_args()
    
    if args.command == "clear":
        clear_dspy_cache_completely()
    elif args.command == "clear-disk":
        clear_dspy_disk_cache(args.cache_dir)
    elif args.command == "reset":
        reset_dspy_cache_settings()
    elif args.command == "status":
        show_cache_status()


if __name__ == "__main__":
    main() 