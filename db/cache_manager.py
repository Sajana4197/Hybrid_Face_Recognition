"""
Automatic cache management system.
Detects when JSONL files change and rebuilds cache automatically.
"""
from pathlib import Path
import time
import threading
import atexit

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = REPO_ROOT / "db"

NH_FILE = DB_DIR / "watchlist_neuralhash.jsonl"
HDIC_FILE = DB_DIR / "watchlist_hdic.jsonl"
CACHE_FILE = DB_DIR / "watchlist_cache.npz"


def cache_exists() -> bool:
    """Check if cache file exists"""
    return CACHE_FILE.exists()


def cache_needs_rebuild() -> bool:
    """
    Check if cache is outdated compared to source JSONL files.
    Returns True if cache should be rebuilt.
    """
    if not CACHE_FILE.exists():
        return True
    
    cache_mtime = CACHE_FILE.stat().st_mtime
    
    # Check if JSONL files are newer than cache
    if NH_FILE.exists() and NH_FILE.stat().st_mtime > cache_mtime:
        return True
    
    if HDIC_FILE.exists() and HDIC_FILE.stat().st_mtime > cache_mtime:
        return True
    
    return False


def rebuild_cache_async():
    """Rebuild cache in background thread (non-blocking)"""
    def _rebuild():
        try:
            from db.build_cache import build_cache
            print("[INFO] 🔄 Rebuilding cache in background...")
            build_cache(silent=True)
            print("[INFO] ✅ Cache rebuild complete")
        except Exception as e:
            print(f"[ERROR] Cache rebuild failed: {e}")
    
    thread = threading.Thread(target=_rebuild, daemon=True)
    thread.start()
    return thread


def rebuild_cache_sync():
    """Rebuild cache synchronously (blocking)"""
    try:
        from db.build_cache import build_cache
        build_cache(silent=False)
        return True
    except Exception as e:
        print(f"[ERROR] Cache rebuild failed: {e}")
        return False


def ensure_cache_exists():
    """
    Ensure cache exists, create if missing.
    This is called at application startup.
    """
    if not cache_exists():
        print("[WARN] Cache not found, building initial cache...")
        return rebuild_cache_sync()
    
    if cache_needs_rebuild():
        print("[INFO] Cache is outdated, rebuilding...")
        return rebuild_cache_sync()
    
    print("[INFO] ✅ Cache is up to date")
    return True


class CacheWatcher:
    """
    Watches for JSONL file changes and automatically rebuilds cache.
    Used by field client to keep cache fresh.
    """
    
    def __init__(self, check_interval=30):
        """
        Args:
            check_interval: How often to check for changes (seconds)
        """
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        self._last_nh_mtime = 0
        self._last_hdic_mtime = 0
        
        # Get initial modification times
        if NH_FILE.exists():
            self._last_nh_mtime = NH_FILE.stat().st_mtime
        if HDIC_FILE.exists():
            self._last_hdic_mtime = HDIC_FILE.stat().st_mtime
    
    def _watch_loop(self):
        """Background thread that watches for file changes"""
        print(f"[INFO] Cache watcher started (checking every {self.check_interval}s)")
        
        while self.running:
            time.sleep(self.check_interval)
            
            try:
                needs_rebuild = False
                
                # Check NH file
                if NH_FILE.exists():
                    nh_mtime = NH_FILE.stat().st_mtime
                    if nh_mtime > self._last_nh_mtime:
                        print("[INFO] NeuralHash watchlist changed")
                        self._last_nh_mtime = nh_mtime
                        needs_rebuild = True
                
                # Check HDIC file
                if HDIC_FILE.exists():
                    hdic_mtime = HDIC_FILE.stat().st_mtime
                    if hdic_mtime > self._last_hdic_mtime:
                        print("[INFO] HDIC watchlist changed")
                        self._last_hdic_mtime = hdic_mtime
                        needs_rebuild = True
                
                # Rebuild if needed
                if needs_rebuild:
                    rebuild_cache_async()
                    
            except Exception as e:
                print(f"[ERROR] Cache watcher error: {e}")
    
    def start(self):
        """Start watching for changes"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._watch_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop watching"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)


# Global watcher instance
_watcher = None


def start_cache_watcher(check_interval=30):
    """
    Start automatic cache watcher.
    Call this in your application startup.
    """
    global _watcher
    if _watcher is None:
        _watcher = CacheWatcher(check_interval)
        _watcher.start()
        # Stop watcher on program exit
        atexit.register(_watcher.stop)
    return _watcher


if __name__ == "__main__":
    if cache_needs_rebuild():
        print("Cache needs rebuild!")
        rebuild_cache_sync()
    else:
        print("Cache is up to date ✅")