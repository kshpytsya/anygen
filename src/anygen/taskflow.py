import collections
import toposort
from .obj import Obj


class TaskFlow:
    def __init__(self):
        self.task_deps = collections.defaultdict(lambda: set())
        self.task_funcs = {}
        self.plan = None

    def add_task(self, name, *, func=None, before=[], after=[]):
        self.plan = None
        assert name not in self.task_funcs
        self.task_funcs[name] = func
        self.task_deps[name]
        for i in after:
            self.task_deps[name].add(i)
        for i in before:
            self.task_deps[i].add(name)

    def add_tasks(self, tasks, first_after=[], last_before=[]):
        next_after = first_after

        for i, (name, func) in enumerate(tasks):
            self.add_task(
                name,
                func=func,
                before=last_before if i == len(tasks) - 1 else [],
                after=next_after
            )

            next_after = [name]

    def _build_plan(self):
        plan = []

        for name in toposort.toposort_flatten(self.task_deps):
            func = self.task_funcs.get(name, None)
            if func is not None:
                plan.append(func)

        self.plan = plan

    def run(self, inp):
        if self.plan is None:
            self._build_plan()

        ctx = Obj(inp=inp, out=None)

        for task in self.plan:
            task(ctx)

        return ctx.out
