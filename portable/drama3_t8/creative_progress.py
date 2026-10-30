"""Compose native frame budgets into fixed logical job progress, never an ETA.

The sink contract is shared by Studio and NativeHost: no arguments check cancel;
integer (value, total) updates progress. Only the host commits the final 100%.
"""
import math


class _Root:
    total = 10000
    def __init__(self, sink):
        self.sink, self.current = sink, 0

    def check(self):
        self.sink()

    def publish(self, value):
        if not math.isfinite(value) or not 0 <= value <= 9900:
            raise ValueError("Invalid composed job progress")
        # These are already mapped whole-job ticks, never raw frame ratios.
        self.current = max(self.current, int(value))
        self.sink(self.current, self.total)


class _Units:
    def __init__(self, root, lower, upper, keys):
        self.root, self.lower, self.upper = root, lower, upper
        self.keys = tuple(keys)
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("Duplicate atomic progress identity")
        self.done, self.active, self.local = set(), None, 0

    def _publish(self, local=0):
        self.root.publish(self.lower + (self.upper-self.lower)*(len(self.done)+local)/len(self.keys))

    def start(self, key):
        self.root.check()
        if key not in self.keys or key in self.done or self.active is not None:
            raise ValueError("Atomic progress does not match the frozen batch plan")
        self.active, self.local = key, 0
        self._publish()
        return self.callback

    def callback(self, value=None, total=None):
        self.root.check()
        if value is None and total is None:
            return
        if (self.active is None or isinstance(value, bool) or isinstance(total, bool)
                or not isinstance(value, (int,float)) or not isinstance(total, (int,float))
                or not math.isfinite(value) or not math.isfinite(total)
                or total <= 0 or not 0 <= value <= total):
            raise ValueError("Invalid native frame progress")
        # Opening retries can restart their frame budget. Their high-water is
        # confined to this one atomic unit. Leave room for decode/commit.
        self.local = max(self.local, min(.9, value/total))
        self._publish(self.local)

    def complete(self, key):
        self.root.check()
        if self.active is None or key != self.active or key in self.done:
            raise ValueError("Atomic completion does not match its active transaction")
        self.done.add(key)
        self.active, self.local = None, 0
        self._publish()

    def reused(self, key):
        self.root.check()
        if key not in self.keys or key == self.active:
            raise ValueError("Reused progress does not match an inactive planned unit")
        if key not in self.done:
            self.done.add(key)
            self._publish()

    def finish(self):
        self.root.check()
        if self.active is not None or len(self.done) != len(self.keys):
            raise ValueError("Unfinished atomic work cannot complete its phase")
        self.root.publish(self.upper)


class _Wait:
    """A real batched result has no per-result callback: hold until it returns."""
    def __init__(self, root, lower, upper):
        self.root, self.lower, self.upper = root, lower, upper
        self.started = self.done = False

    def start(self):
        self.root.check()
        if self.started or self.done:
            raise ValueError("Quality progress phase already started")
        self.started = True
        self.root.publish(self.lower)
        return self.check

    def check(self, *args):
        self.root.check()

    def finish(self, *, skipped=False):
        self.root.check()
        if self.done or (not self.started and not skipped):
            raise ValueError("Quality phase has no completed batch")
        self.done = True
        self.root.publish(self.upper)


class _RetryRound:
    def __init__(self, root, lower, upper, source_ids):
        boundary = lower + (upper-lower)*.8
        self.generation = _Units(root, lower, boundary, source_ids)
        self.quality = _Wait(root, boundary, upper)

    def start(self, source_id):
        return self.generation.start(source_id)

    def complete(self, source_id):
        self.generation.complete(source_id)

    def generation_done(self):
        self.generation.finish()


class RetryProgress:
    def __init__(self, root, limit):
        self.root, self.limit, self.rounds = root, limit, {}
        self.skipped_rounds, self.done = [], limit == 0

    def round(self, number, source_ids):
        self.root.check()
        if (type(number) is not int or not 1 <= number <= self.limit or number != len(self.rounds)+1
                or self.done or self.rounds and not self.rounds[number-1].quality.done):
            raise ValueError("Invalid bounded retry progress round")
        width = 2800/self.limit
        scope = _RetryRound(self.root, 7000+(number-1)*width, 7000+number*width, tuple(source_ids))
        self.rounds[number] = scope
        return scope

    def finish(self):
        self.root.check()
        if self.done:
            return
        for number in range(1,self.limit+1):
            scope = self.rounds.get(number)
            if scope is None:
                # Called only after the real controller establishes no pending
                # eligible source; these are skipped slots, not fake attempts.
                self.skipped_rounds.append(number)
            elif (scope.generation.active is not None
                    or len(scope.generation.done) != len(scope.generation.keys) or not scope.quality.done):
                raise ValueError("Retry progress still has unfinished real work")
        self.done = True
        self.root.publish(9800)


class BatchProgress:
    """Stable atomic generation, one QA batch, at most two retake rounds.

    Phase weights are logical UI allocation, not elapsed-time estimates. Slots
    never expand after starting. Dependency-blocked jobs are reserved up front;
    intact reused/locked/not-requested units do not count as new work.
    """
    def __init__(self, sink, plan, *, quality=False, retries=0):
        if type(retries) is not int or not 0 <= retries <= 2:
            raise ValueError("Progress supports zero to two quality retries")
        self.root = _Root(sink)
        quality = bool(quality or retries)
        end = 6000 if retries else 8000 if quality else 9800
        self.generation = _Units(self.root,0,end,[self.key(job) for job in plan["jobs"]
            if job["status"] in {"pending","blocked"}])
        self.quality = _Wait(self.root,end,7000 if retries else 9800) if quality else None
        self.retries = RetryProgress(self.root,retries)
        self.attempted = 0
        self.reused_keys = set()

    @staticmethod
    def key(job):
        return job["unit_id"], job["take_index"]

    def check(self, *args):
        self.root.check()

    def start(self, job):
        return self.generation.start(self.key(job))

    def updated(self, record):
        count = record["generated"] + record["failed"]
        if count != self.attempted+1:
            raise ValueError("Workspace progress did not follow one committed atomic result")
        self.generation.complete(self.generation.active)
        self.attempted = count
        self._reconcile(record["plan"])

    def _reconcile(self, plan):
        # A restored earlier selection can make an initially blocked later
        # candidate reusable. Settle its fixed slot without inventing a new
        # generation callback or mixing it into this run's attempt count.
        for job in plan["jobs"]:
            key = self.key(job)
            if key in self.generation.keys and key not in self.generation.done and job["status"] == "completed":
                self.generation.reused(key)
                self.reused_keys.add(key)

    def generation_done(self, plan=None):
        if plan is not None:self._reconcile(plan)
        self.generation.finish()

    def commit(self):
        self.root.check()
        if (self.quality is not None and not self.quality.done) or not self.retries.done:
            raise ValueError("Quality work is unfinished before commit")
        self.root.publish(9800)

    def committed(self):
        self.root.publish(9900)
