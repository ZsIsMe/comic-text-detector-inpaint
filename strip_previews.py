"""Load the visible non-editing pages without blocking the Qt event loop."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import logging

from PySide6.QtCore import QObject, Qt, Signal, Slot


class StripPreviews(QObject):
    _completed = Signal(int, object, object)

    # In addition to the visible pages, retain a small budget for scrolling back.
    _SPARE_PAGES = 2
    _SPARE_BYTES = 64 * 1024 * 1024

    def __init__(self, owner, load_func, apply_func):
        super().__init__(owner)
        self._load = load_func
        self._apply = apply_func
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='strip-preview')
        self._completed.connect(self._on_completed, Qt.ConnectionType.QueuedConnection)
        self._settings = None
        self._generation = 0
        self._requested = ()
        self._wanted = set()
        self._cache = OrderedDict()
        self._failed = set()
        self._published = frozenset()
        self._running_path = None
        self._running_invalidated = False
        self._closed = False

    def request(self, paths, settings, *, ready=None):
        """Request visible paths, optionally supplying fresh previews to show immediately."""
        if self._closed:
            return
        paths = tuple(dict.fromkeys(paths))
        if paths == self._requested and settings == self._settings and not ready:
            return
        self._requested = paths
        self._wanted = set(paths)
        self._failed.intersection_update(self._wanted)
        if settings != self._settings:
            self._settings = settings
            self._discard_results()
        seeded = False
        for path, pair in (ready or {}).items():
            if path in self._wanted:
                self._cache[path] = pair
                self._failed.discard(path)
                if self._running_path == path:
                    self._running_invalidated = True
                seeded = True
        self._refresh(republish=seeded)

    def invalidate(self):
        """Forget previous results and rebuild any views that are still visible."""
        if self._closed:
            return
        self._discard_results()
        self._refresh()

    def _discard_results(self):
        self._generation += 1
        self._cache.clear()
        self._failed.clear()

    def invalidate_path(self, path):
        """Drop both a cached preview and any pending read of this page."""
        if self._closed:
            return
        self._cache.pop(path, None)
        self._failed.discard(path)
        if self._running_path == path:
            self._running_invalidated = True
        self._refresh()

    def _refresh(self, *, republish=False):
        for path in self._requested:
            if path in self._cache:
                self._cache.move_to_end(path)

        # Newest spare entries get the budget first. A very large page can
        # remain visible but will be released immediately after it leaves view.
        spare_count = spare_bytes = 0
        for path in reversed(tuple(self._cache)):
            if path in self._wanted:
                continue
            size = sum(image.sizeInBytes() for image in self._cache[path])
            if spare_count >= self._SPARE_PAGES or spare_bytes + size > self._SPARE_BYTES:
                del self._cache[path]
            else:
                spare_count += 1
                spare_bytes += size

        visible = {p: self._cache[p] for p in self._requested if p in self._cache}
        keys = frozenset(visible)
        if republish or keys != self._published:
            self._published = keys
            self._apply(visible)
        if self._running_path is not None or self._closed:
            return
        path = next((p for p in self._requested if p not in self._cache and p not in self._failed), None)
        if path is None:
            return
        self._running_path = path
        self._running_invalidated = False
        generation = self._generation
        job = self._executor.submit(self._load, path, self._settings)

        def done(future):
            if not self._closed:
                try:
                    self._completed.emit(generation, path, future)
                except RuntimeError:
                    # The owner can be deleted during application shutdown.
                    pass

        job.add_done_callback(done)

    @Slot(int, object, object)
    def _on_completed(self, generation, path, future):
        if self._closed:
            return
        self._running_path = None
        if generation == self._generation and not self._running_invalidated and path in self._wanted:
            try:
                pair = future.result()
            except Exception:
                logging.getLogger(__name__).exception('Unable to load page previews for %s', path)
                pair = None
            if pair is None:
                self._failed.add(path)
            else:
                self._cache[path] = pair
        self._refresh()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._cache.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
