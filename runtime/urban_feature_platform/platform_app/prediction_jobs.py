"""A single GPU worker with bounded local job state and progress polling."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import threading
import time
import uuid


class PredictionJobs:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="urban-inference")
        self._lock = threading.Lock()
        self._jobs = {}

    def submit(self, function):
        identifier = uuid.uuid4().hex
        with self._lock:
            active = sum(job["status"] in {"queued", "running"} for job in self._jobs.values())
            if active >= 3:
                raise ValueError("The inference queue is full. Please wait for the current predictions.")
            for key in list(self._jobs):
                if len(self._jobs) < 32:
                    break
                if self._jobs[key]["status"] in {"complete", "failed"}:
                    del self._jobs[key]
            self._jobs[identifier] = {"job_id": identifier, "status": "queued", "progress": 0,
                                      "message": "Queued for the local model", "created_at": datetime.now().isoformat(),
                                      "_submitted_clock": time.perf_counter(), "_started_clock": None, "_finished_clock": None}

        def update(fraction, message):
            with self._lock:
                self._jobs[identifier].update(status="running", progress=round(max(0., min(1., fraction)) * 100, 1), message=message)

        def run():
            try:
                with self._lock:
                    self._jobs[identifier]["_started_clock"] = time.perf_counter()
                update(0, "Checking input and loading the selected model")
                result = function(update)
                with self._lock:
                    job = self._jobs[identifier]
                    job["_finished_clock"] = time.perf_counter()
                    timing = {"queue_seconds": job["_started_clock"] - job["_submitted_clock"],
                              "worker_seconds": job["_finished_clock"] - job["_started_clock"],
                              "submitted_to_complete_seconds": job["_finished_clock"] - job["_submitted_clock"]}
                    if isinstance(result, dict):
                        result["job_timing"] = timing
                    job.update(status="complete", progress=100, message="Prediction complete", result=result)
            except Exception as error:
                with self._lock:
                    self._jobs[identifier].update(status="failed", error=str(error), message=str(error), _finished_clock=time.perf_counter())

        self._executor.submit(run)
        return {"job_id": identifier, "status": "queued", "status_url": f"/api/jobs/{identifier}"}

    def get(self, identifier):
        with self._lock:
            if identifier not in self._jobs:
                raise KeyError("Prediction job not found. If the app restarted, submit the image again.")
            job = self._jobs[identifier]
            value = {key: item for key, item in job.items() if not key.startswith("_")}
            now = job["_finished_clock"] or time.perf_counter()
            value["elapsed_seconds"] = max(0., now - job["_submitted_clock"])
            value["queue_seconds"] = max(0., (job["_started_clock"] or now) - job["_submitted_clock"])
            value["running_seconds"] = max(0., now - job["_started_clock"]) if job["_started_clock"] else 0.
            return value
