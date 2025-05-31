import wget
import sys
import time

def custom_progress_bar():
    start_time = time.time()

    def progress(current, total, width=50):
        elapsed = time.time() - start_time
        speed = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / speed if speed > 0 else 0

        downloaded_mb = current / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        percent = current / total * 100
        bar = '#' * int(width * current / total)

        eta_str = time.strftime("%M:%S", time.gmtime(eta))

        sys.stdout.write(f"\r[{bar:<{width}}] {downloaded_mb:.2f}MB / {total_mb:.2f}MB ({percent:.1f}%) ETA: {eta_str}")
        sys.stdout.flush()

    return progress
